#ifdef WITH_XPU

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/types.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "voxelization.h"

namespace voxelization {
namespace {
constexpr int kThreadsPerBlock = 512;
constexpr int kMaxGridDim = 50000;

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

inline sycl::nd_range<1> make_launch(int total_work_items) {
  if (total_work_items <= 0) {
    return sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1));
  }
  const int groups = std::max(1, std::min(ceil_div(total_work_items, kThreadsPerBlock), kMaxGridDim));
  const int global = groups * kThreadsPerBlock;
  return sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreadsPerBlock));
}

template <typename T>
inline void atomic_add(T *address, T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*address);
  ref.fetch_add(value);
}

template <typename T>
inline void atomic_max(T *address, T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*address);
  T old = ref.load();
  while (old < value && !ref.compare_exchange_strong(old, value)) {
  }
}

template <typename T>
inline void atomic_min(T *address, T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*address);
  T old = ref.load();
  while (old > value && !ref.compare_exchange_strong(old, value)) {
  }
}

template <typename T>
void feats_reduce_kernel(sycl::queue &queue, const T *feats,
                         const int32_t *coors_map, T *reduced_feats,
                         int num_input, int num_feats, reduce_t reduce_type) {
  if (num_input <= 0) {
    return;
  }
  auto launch = make_launch(num_input);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int x = item.get_global_linear_id();
      if (x >= num_input) {
        return;
      }
      const int32_t reduce_to = coors_map[x];
      if (reduce_to == -1) {
        return;
      }
      const T *feats_offset = feats + x * num_feats;
      T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;
      if (reduce_type == reduce_t::MAX) {
        for (int i = 0; i < num_feats; ++i) {
          atomic_max(reduced_feats_offset + i, feats_offset[i]);
        }
      } else {
        for (int i = 0; i < num_feats; ++i) {
          atomic_add(reduced_feats_offset + i, feats_offset[i]);
        }
      }
    });
  });
}

template <typename T>
void add_reduce_traceback_grad_kernel(sycl::queue &queue, T *grad_feats,
                                      const T *grad_reduced_feats,
                                      const int32_t *coors_map,
                                      const int32_t *reduce_count,
                                      int num_input, int num_feats,
                                      reduce_t reduce_type) {
  if (num_input <= 0) {
    return;
  }
  auto launch = make_launch(num_input);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int x = item.get_global_linear_id();
      if (x >= num_input) {
        return;
      }
      const int32_t reduce_to = coors_map[x];
      if (reduce_to == -1) {
        return;
      }
      const int input_offset = x * num_feats;
      T *grad_feats_offset = grad_feats + input_offset;
      const int reduced_offset = reduce_to * num_feats;
      const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;
      if (reduce_type == reduce_t::SUM) {
        for (int i = 0; i < num_feats; ++i) {
          grad_feats_offset[i] = grad_reduced_feats_offset[i];
        }
      } else if (reduce_type == reduce_t::MEAN) {
        const T denom = static_cast<T>(reduce_count[reduce_to]);
        for (int i = 0; i < num_feats; ++i) {
          grad_feats_offset[i] = grad_reduced_feats_offset[i] / denom;
        }
      }
    });
  });
}

template <typename T>
void max_reduce_traceback_scatter_idx_kernel(
    sycl::queue &queue, const T *feats, const T *reduced_feats,
    int32_t *reduce_from, const int32_t *coors_map, int num_input,
    int num_feats) {
  if (num_input <= 0) {
    return;
  }
  auto launch = make_launch(num_input);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int x = item.get_global_linear_id();
      if (x >= num_input) {
        return;
      }
      const int32_t reduce_to = coors_map[x];
      if (reduce_to == -1) {
        return;
      }
      const int input_offset = x * num_feats;
      const T *feats_offset = feats + input_offset;
      const int reduced_offset = reduce_to * num_feats;
      const T *reduced_feats_offset = reduced_feats + reduced_offset;
      int32_t *reduce_from_offset = reduce_from + reduced_offset;
      for (int i = 0; i < num_feats; ++i) {
        if (feats_offset[i] == reduced_feats_offset[i]) {
          atomic_min(reduce_from_offset + i, static_cast<int32_t>(x));
        }
      }
    });
  });
}

template <typename T>
void max_reduce_scatter_grad_kernel(sycl::queue &queue, T *grad_feats,
                                    const T *grad_reduced_feats,
                                    const int32_t *reduce_from,
                                    int num_reduced, int num_feats) {
  if (num_reduced <= 0) {
    return;
  }
  auto launch = make_launch(num_reduced);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int x = item.get_global_linear_id();
      if (x >= num_reduced) {
        return;
      }
      const int reduced_offset = x * num_feats;
      const int32_t *scatter_to_offset = reduce_from + reduced_offset;
      const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;
      for (int i = 0; i < num_feats; ++i) {
        grad_feats[scatter_to_offset[i] * num_feats + i] =
            grad_reduced_feats_offset[i];
      }
    });
  });
}

}  // namespace

std::vector<at::Tensor> dynamic_point_to_voxel_forward_xpu(
    const at::Tensor &feats, const at::Tensor &coors,
    reduce_t reduce_type) {
  TORCH_CHECK(feats.device().type() == c10::DeviceType::XPU,
              "feats must be on XPU for dynamic_point_to_voxel_forward_xpu");
  TORCH_CHECK(coors.device().type() == c10::DeviceType::XPU,
              "coors must be on XPU for dynamic_point_to_voxel_forward_xpu");

  c10::OptionalDeviceGuard guard(feats.device());
  auto &queue = c10::xpu::getCurrentXPUStream().queue();

  const int num_input = feats.size(0);
  const int num_feats = feats.size(1);

  if (num_input == 0) {
    return {feats.clone().detach(), coors.clone().detach(),
            coors.new_empty({0}, torch::kInt32),
            coors.new_empty({0}, torch::kInt32)};
  }

  at::Tensor out_coors;
  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);

  std::tie(out_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, true);

  if (out_coors.size(0) > 0 && out_coors.index({0, 0}).lt(0).item<bool>()) {
    out_coors = out_coors.slice(0, 1);
    reduce_count = reduce_count.slice(0, 1);
    coors_map = coors_map - 1;
  }

  coors_map = coors_map.to(torch::kInt32);
  reduce_count = reduce_count.to(torch::kInt32);

  auto reduced_feats =
      at::empty({out_coors.size(0), num_feats}, feats.options());

  AT_DISPATCH_FLOATING_TYPES(
      feats.scalar_type(), "dynamic_point_to_voxel_forward_xpu", ([&] {
        if (reduce_type == reduce_t::MAX) {
          reduced_feats.fill_(-std::numeric_limits<scalar_t>::infinity());
        } else {
          reduced_feats.fill_(static_cast<scalar_t>(0));
        }
        feats_reduce_kernel<scalar_t>(
            queue, feats.data_ptr<scalar_t>(),
            coors_map.data_ptr<int32_t>(),
            reduced_feats.data_ptr<scalar_t>(), num_input, num_feats,
            reduce_type);
        queue.wait_and_throw();
        if (reduce_type == reduce_t::MEAN) {
          reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
        }
      }));

  return {reduced_feats, out_coors, coors_map, reduce_count};
}

void dynamic_point_to_voxel_backward_xpu(
    at::Tensor &grad_feats, const at::Tensor &grad_reduced_feats,
    const at::Tensor &feats, const at::Tensor &reduced_feats,
    const at::Tensor &coors_map, const at::Tensor &reduce_count,
    reduce_t reduce_type) {
  TORCH_CHECK(grad_feats.device().type() == c10::DeviceType::XPU,
              "grad_feats must be on XPU for dynamic_point_to_voxel_backward_xpu");
  TORCH_CHECK(grad_reduced_feats.device().type() == c10::DeviceType::XPU,
              "grad_reduced_feats must be on XPU for dynamic_point_to_voxel_backward_xpu");
  TORCH_CHECK(feats.device().type() == c10::DeviceType::XPU,
              "feats must be on XPU for dynamic_point_to_voxel_backward_xpu");
  TORCH_CHECK(reduced_feats.device().type() == c10::DeviceType::XPU,
              "reduced_feats must be on XPU for dynamic_point_to_voxel_backward_xpu");
  TORCH_CHECK(coors_map.device().type() == c10::DeviceType::XPU,
              "coors_map must be on XPU for dynamic_point_to_voxel_backward_xpu");
  TORCH_CHECK(reduce_count.device().type() == c10::DeviceType::XPU,
              "reduce_count must be on XPU for dynamic_point_to_voxel_backward_xpu");

  c10::OptionalDeviceGuard guard(grad_feats.device());
  auto &queue = c10::xpu::getCurrentXPUStream().queue();

  const int num_input = feats.size(0);
  const int num_reduced = reduced_feats.size(0);
  const int num_feats = feats.size(1);

  grad_feats.fill_(0);

  if (num_input == 0 || num_reduced == 0) {
    return;
  }

  if (reduce_type == reduce_t::MEAN || reduce_type == reduce_t::SUM) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "dynamic_point_to_voxel_backward_add_xpu", ([&] {
          add_reduce_traceback_grad_kernel<scalar_t>(
              queue, grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              coors_map.data_ptr<int32_t>(),
              reduce_count.data_ptr<int32_t>(), num_input, num_feats,
              reduce_type);
          queue.wait_and_throw();
        }));
  } else {
    auto reduce_from = at::full({num_reduced, num_feats}, num_input,
                                coors_map.options().dtype(torch::kInt32));

    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "dynamic_point_to_voxel_backward_max_idx_xpu", ([&] {
          max_reduce_traceback_scatter_idx_kernel<scalar_t>(
              queue, feats.data_ptr<scalar_t>(),
              reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(),
              coors_map.data_ptr<int32_t>(), num_input, num_feats);
          queue.wait_and_throw();
          max_reduce_scatter_grad_kernel<scalar_t>(
              queue, grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(), num_reduced, num_feats);
          queue.wait_and_throw();
        }));
  }
}

}  // namespace voxelization

#endif  // WITH_XPU
