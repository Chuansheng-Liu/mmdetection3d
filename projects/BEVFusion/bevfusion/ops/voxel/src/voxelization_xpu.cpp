#ifdef WITH_XPU

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "voxelization.h"

namespace voxelization {
namespace {
constexpr int kThreadsPerBlock = 512;

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

inline sycl::nd_range<1> make_launch_config(int total_work_items) {
  const int groups = std::max(1, ceil_div(total_work_items, kThreadsPerBlock));
  const int global = groups * kThreadsPerBlock;
  return sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreadsPerBlock));
}

template <typename T, typename IndexT>
void dynamic_voxelize_kernel(sycl::queue &queue, const T *points,
                             IndexT *coors, float voxel_x, float voxel_y,
                             float voxel_z, float coors_x_min,
                             float coors_y_min, float coors_z_min,
                             float coors_x_max, float coors_y_max,
                             float coors_z_max, int grid_x, int grid_y,
                             int grid_z, int num_points, int num_features,
                             int ndim) {
  if (num_points <= 0) {
    return;
  }
  auto launch = make_launch_config(num_points);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int index = item.get_global_linear_id();
      if (index >= num_points) {
        return;
      }
      const T *points_offset = points + index * num_features;
      IndexT *coors_offset = coors + index * ndim;
      const int cx = static_cast<int>(sycl::floor((points_offset[0] - coors_x_min) / voxel_x));
      if (cx < 0 || cx >= grid_x) {
        coors_offset[0] = -1;
        return;
      }
      const int cy = static_cast<int>(sycl::floor((points_offset[1] - coors_y_min) / voxel_y));
      if (cy < 0 || cy >= grid_y) {
        coors_offset[0] = -1;
        coors_offset[1] = -1;
        return;
      }
      const int cz = static_cast<int>(sycl::floor((points_offset[2] - coors_z_min) / voxel_z));
      if (cz < 0 || cz >= grid_z) {
        coors_offset[0] = -1;
        coors_offset[1] = -1;
        coors_offset[2] = -1;
        return;
      }
      coors_offset[0] = cx;
      coors_offset[1] = cy;
      coors_offset[2] = cz;
    });
  });
}

template <typename T, typename IndexT>
void assign_point_to_voxel_kernel(sycl::queue &queue, int nthreads,
                                  const T *points, IndexT *point_to_voxelidx,
                                  IndexT *coor_to_voxelidx, T *voxels,
                                  int max_points, int num_features,
                                  int num_points, int ndim) {
  if (nthreads <= 0) {
    return;
  }
  auto launch = make_launch_config(nthreads);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int thread_idx = item.get_global_linear_id();
      if (thread_idx >= nthreads) {
        return;
      }
      const int index = thread_idx / num_features;
      const int num = point_to_voxelidx[index];
      const int voxelidx = coor_to_voxelidx[index];
      if (num > -1 && voxelidx > -1) {
        T *voxels_offset =
            voxels + voxelidx * max_points * num_features + num * num_features;
        const int k = thread_idx % num_features;
        voxels_offset[k] = points[thread_idx];
      }
    });
  });
}

template <typename IndexT>
void assign_voxel_coors_kernel(sycl::queue &queue, int nthreads, IndexT *coor,
                               IndexT *point_to_voxelidx,
                               IndexT *coor_to_voxelidx,
                               IndexT *voxel_coors, int num_points,
                               int ndim) {
  if (nthreads <= 0) {
    return;
  }
  auto launch = make_launch_config(nthreads);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int thread_idx = item.get_global_linear_id();
      if (thread_idx >= nthreads) {
        return;
      }
      const int index = thread_idx / ndim;
      const int num = point_to_voxelidx[index];
      const int voxelidx = coor_to_voxelidx[index];
      if (num == 0 && voxelidx > -1) {
        IndexT *coors_offset = voxel_coors + voxelidx * ndim;
        const int k = thread_idx % ndim;
        coors_offset[k] = coor[thread_idx];
      }
    });
  });
}

template <typename IndexT>
void point_to_voxelidx_kernel(sycl::queue &queue, const IndexT *coor,
                              IndexT *point_to_voxelidx,
                              IndexT *point_to_pointidx, int max_points,
                              int max_voxels, int num_points, int ndim) {
  (void)max_voxels;
  if (num_points <= 0) {
    return;
  }
  auto launch = make_launch_config(num_points);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int index = item.get_global_linear_id();
      if (index >= num_points) {
        return;
      }
      const IndexT *coor_offset = coor + index * ndim;
      if (coor_offset[0] == -1) {
        return;
      }
      int num = 0;
      const int coor_x = coor_offset[0];
      const int coor_y = coor_offset[1];
      const int coor_z = coor_offset[2];
      for (int i = 0; i < index; ++i) {
        const IndexT *prev_coor = coor + i * ndim;
        if (prev_coor[0] == -1) {
          continue;
        }
        if (prev_coor[0] == coor_x && prev_coor[1] == coor_y &&
            prev_coor[2] == coor_z) {
          ++num;
          if (num == 1) {
            point_to_pointidx[index] = i;
          } else if (num >= max_points) {
            return;
          }
        }
      }
      if (num == 0) {
        point_to_pointidx[index] = index;
      }
      if (num < max_points) {
        point_to_voxelidx[index] = num;
      }
    });
  });
}

template <typename IndexT>
void determin_voxel_num_kernel(sycl::queue &queue,
                               IndexT *num_points_per_voxel,
                               IndexT *point_to_voxelidx,
                               IndexT *point_to_pointidx,
                               IndexT *coor_to_voxelidx, IndexT *voxel_num,
                               int max_points, int max_voxels, int num_points) {
  auto launch = sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1));
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1>) {
      for (int i = 0; i < num_points; ++i) {
        const int point_pos_in_voxel = point_to_voxelidx[i];
        if (point_pos_in_voxel == -1) {
          continue;
        } else if (point_pos_in_voxel == 0) {
          IndexT voxelidx = voxel_num[0];
          if (voxel_num[0] >= max_voxels) {
            continue;
          }
          voxel_num[0] += 1;
          coor_to_voxelidx[i] = voxelidx;
          num_points_per_voxel[voxelidx] = 1;
        } else {
          const int point_idx = point_to_pointidx[i];
          const IndexT voxelidx = coor_to_voxelidx[point_idx];
          if (voxelidx != -1) {
            coor_to_voxelidx[i] = voxelidx;
            num_points_per_voxel[voxelidx] += 1;
          }
        }
      }
    });
  });
}

template <typename IndexT>
void nondisterministic_get_assign_pos_kernel(
    sycl::queue &queue, int nthreads, const IndexT *coors_map, IndexT *pts_id,
    IndexT *coors_count, IndexT *reduce_count, IndexT *coors_order) {
  if (nthreads <= 0) {
    return;
  }
  auto launch = make_launch_config(nthreads);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int thread_idx = item.get_global_linear_id();
      if (thread_idx >= nthreads) {
        return;
      }
      const int coors_idx = coors_map[thread_idx];
      if (coors_idx > -1) {
        sycl::atomic_ref<IndexT, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            reduce_ref(reduce_count[coors_idx]);
        IndexT coors_pts_pos = reduce_ref.fetch_add(1);
        pts_id[thread_idx] = coors_pts_pos;
        if (coors_pts_pos == 0) {
          sycl::atomic_ref<IndexT, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              count_ref(coors_count[0]);
          IndexT order = count_ref.fetch_add(1);
          coors_order[coors_idx] = order;
        }
      }
    });
  });
}

template <typename T, typename IndexT>
void nondisterministic_assign_point_voxel_kernel(
    sycl::queue &queue, int nthreads, const T *points,
    const IndexT *coors_map, const IndexT *pts_id, const IndexT *coors_in,
    const IndexT *reduce_count, const IndexT *coors_order, T *voxels,
    IndexT *coors, IndexT *pts_count, int max_voxels, int max_points,
    int num_features, int ndim) {
  if (nthreads <= 0) {
    return;
  }
  auto launch = make_launch_config(nthreads);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int thread_idx = item.get_global_linear_id();
      if (thread_idx >= nthreads) {
        return;
      }
      const int coors_idx = coors_map[thread_idx];
      const int coors_pts_pos = pts_id[thread_idx];
      if (coors_idx > -1) {
        const int coors_pos = coors_order[coors_idx];
        if (coors_pos < max_voxels && coors_pts_pos < max_points) {
          T *voxels_offset = voxels +
                             (coors_pos * max_points + coors_pts_pos) *
                                 num_features;
          const T *points_offset = points + thread_idx * num_features;
          for (int k = 0; k < num_features; ++k) {
            voxels_offset[k] = points_offset[k];
          }
          if (coors_pts_pos == 0) {
      pts_count[coors_pos] = sycl::min(
        reduce_count[coors_idx], static_cast<IndexT>(max_points));
            IndexT *coors_offset = coors + coors_pos * ndim;
            const IndexT *coors_in_offset = coors_in + coors_idx * ndim;
            for (int k = 0; k < ndim; ++k) {
              coors_offset[k] = coors_in_offset[k];
            }
          }
        }
      }
    });
  });
}

}  // namespace

int hard_voxelize_xpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      int max_points, int max_voxels, int ndim) {
  TORCH_CHECK(points.device().type() == c10::DeviceType::XPU,
              "points must be on XPU for hard_voxelize_xpu");
  c10::OptionalDeviceGuard guard(points.device());
  auto &queue = c10::xpu::getCurrentXPUStream().queue();

  const int num_points = points.size(0);
  const int num_features = points.size(1);
  if (num_points == 0) {
    return 0;
  }

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = static_cast<int>(std::round((coors_x_max - coors_x_min) / voxel_x));
  const int grid_y = static_cast<int>(std::round((coors_y_max - coors_y_min) / voxel_y));
  const int grid_z = static_cast<int>(std::round((coors_z_max - coors_z_min) / voxel_z));

  at::Tensor temp_coors =
      at::zeros({num_points, ndim}, points.options().dtype(at::kInt));

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "hard_voxelize_xpu", ([&] {
    dynamic_voxelize_kernel<scalar_t, int>(
        queue, points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, ndim);
  }));
  queue.wait_and_throw();

  at::Tensor point_to_pointidx = -at::ones({num_points}, points.options().dtype(at::kInt));
  at::Tensor point_to_voxelidx = -at::ones({num_points}, points.options().dtype(at::kInt));

  point_to_voxelidx_kernel<int>(
      queue, temp_coors.contiguous().data_ptr<int>(),
      point_to_voxelidx.contiguous().data_ptr<int>(),
      point_to_pointidx.contiguous().data_ptr<int>(), max_points, max_voxels,
      num_points, ndim);
  queue.wait_and_throw();

  at::Tensor coor_to_voxelidx = -at::ones({num_points}, points.options().dtype(at::kInt));
  at::Tensor voxel_num = at::zeros({1}, points.options().dtype(at::kInt));

  determin_voxel_num_kernel<int>(
      queue, num_points_per_voxel.contiguous().data_ptr<int>(),
      point_to_voxelidx.contiguous().data_ptr<int>(),
      point_to_pointidx.contiguous().data_ptr<int>(),
      coor_to_voxelidx.contiguous().data_ptr<int>(),
      voxel_num.contiguous().data_ptr<int>(), max_points, max_voxels,
      num_points);
  queue.wait_and_throw();

  const int pts_output_size = num_points * num_features;
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_point_to_voxel_xpu", ([&] {
    assign_point_to_voxel_kernel<float, int>(
        queue, pts_output_size, points.contiguous().data_ptr<float>(),
        point_to_voxelidx.contiguous().data_ptr<int>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxels.contiguous().data_ptr<float>(), max_points, num_features,
        num_points, ndim);
  }));

  const int coors_output_size = num_points * ndim;
  assign_voxel_coors_kernel<int>(
    queue, coors_output_size, temp_coors.contiguous().data_ptr<int>(),
      point_to_voxelidx.contiguous().data_ptr<int>(),
      coor_to_voxelidx.contiguous().data_ptr<int>(),
      coors.contiguous().data_ptr<int>(), num_points, ndim);
  queue.wait_and_throw();

  at::Tensor voxel_num_cpu = voxel_num.to(at::kCPU);
  return voxel_num_cpu.data_ptr<int>()[0];
}

int nondisterministic_hard_voxelize_xpu(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, int max_points, int max_voxels,
    int ndim) {
  TORCH_CHECK(points.device().type() == c10::DeviceType::XPU,
              "points must be on XPU for nondisterministic_hard_voxelize_xpu");
  c10::OptionalDeviceGuard guard(points.device());
  auto &queue = c10::xpu::getCurrentXPUStream().queue();

  const int num_points = points.size(0);
  const int num_features = points.size(1);
  if (num_points == 0) {
    return 0;
  }

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = static_cast<int>(std::round((coors_x_max - coors_x_min) / voxel_x));
  const int grid_y = static_cast<int>(std::round((coors_y_max - coors_y_min) / voxel_y));
  const int grid_z = static_cast<int>(std::round((coors_z_max - coors_z_min) / voxel_z));

  at::Tensor temp_coors =
      at::zeros({num_points, ndim}, points.options().dtype(torch::kInt32));

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "nondet_hard_voxelize_xpu", ([&] {
    dynamic_voxelize_kernel<scalar_t, int>(
        queue, points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, ndim);
  }));
  queue.wait_and_throw();

  at::Tensor coors_clean =
      temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  at::Tensor coors_map;
  at::Tensor reduce_count;
  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors.size(0) > 0 && temp_coors.index({0, 0}).lt(0).item<bool>()) {
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  const int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(torch::kInt32);
  coors_map = coors_map.to(torch::kInt32);

  at::Tensor coors_count = coors_map.new_zeros({1});
  at::Tensor coors_order = coors_map.new_empty({num_coors});
  reduce_count = reduce_count.to(torch::kInt32);
  at::Tensor pts_id = coors_map.new_zeros({num_points});

  nondisterministic_get_assign_pos_kernel<int>(
      queue, num_points, coors_map.contiguous().data_ptr<int>(),
      pts_id.contiguous().data_ptr<int>(),
      coors_count.contiguous().data_ptr<int>(),
      reduce_count.contiguous().data_ptr<int>(),
      coors_order.contiguous().data_ptr<int>());

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "assign_point_voxel_nondet_xpu",
                        ([&] {
                          nondisterministic_assign_point_voxel_kernel<scalar_t, int>(
                              queue, num_points,
                              points.contiguous().data_ptr<scalar_t>(),
                              coors_map.contiguous().data_ptr<int>(),
                              pts_id.contiguous().data_ptr<int>(),
                              temp_coors.contiguous().data_ptr<int>(),
                              reduce_count.contiguous().data_ptr<int>(),
                              coors_order.contiguous().data_ptr<int>(),
                              voxels.contiguous().data_ptr<scalar_t>(),
                              coors.contiguous().data_ptr<int>(),
                              num_points_per_voxel.contiguous().data_ptr<int>(),
                              max_voxels, max_points, num_features, ndim);
                        }));
  queue.wait_and_throw();

  return std::min(max_voxels, num_coors);
}

void dynamic_voxelize_xpu(const at::Tensor &points, at::Tensor &coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range, int ndim) {
  TORCH_CHECK(points.device().type() == c10::DeviceType::XPU,
              "points must be on XPU for dynamic_voxelize_xpu");
  c10::OptionalDeviceGuard guard(points.device());
  auto &queue = c10::xpu::getCurrentXPUStream().queue();

  const int num_points = points.size(0);
  const int num_features = points.size(1);
  if (num_points == 0) {
    return;
  }

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = static_cast<int>(std::round((coors_x_max - coors_x_min) / voxel_x));
  const int grid_y = static_cast<int>(std::round((coors_y_max - coors_y_min) / voxel_y));
  const int grid_z = static_cast<int>(std::round((coors_z_max - coors_z_min) / voxel_z));

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_xpu", ([&] {
    dynamic_voxelize_kernel<scalar_t, int>(
        queue, points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, ndim);
  }));
  queue.wait_and_throw();
}

}  // namespace voxelization

#endif  // WITH_XPU
