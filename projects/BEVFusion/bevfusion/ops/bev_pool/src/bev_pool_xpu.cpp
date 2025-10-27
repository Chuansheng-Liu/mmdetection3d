#include <sycl/sycl.hpp>

#include <algorithm>

#include <c10/xpu/XPUStream.h>

namespace {
constexpr int kWorkGroupSize = 256;

inline int ceil_div(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

inline sycl::nd_range<1> make_1d_launch(int total_work_items) {
  const int groups = std::max(1, ceil_div(total_work_items, kWorkGroupSize));
  const int global = groups * kWorkGroupSize;
  return sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize));
}
}  // namespace

void bev_pool_xpu(int b, int d, int h, int w, int n, int c, int n_intervals,
                  const float* x, const int* geom_feats,
                  const int* interval_starts, const int* interval_lengths,
                  float* out) {
  (void)b;
  (void)n;
  if (n_intervals == 0 || c == 0) {
    return;
  }

  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int total_work_items = n_intervals * c;
  auto launch = make_1d_launch(total_work_items);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class BevPoolForward>(launch, [=](sycl::nd_item<1> item) {
      const int idx = item.get_global_linear_id();
      if (idx >= total_work_items) {
        return;
      }
      const int interval_index = idx / c;
      const int channel_index = idx % c;
      if (interval_index >= n_intervals) {
        return;
      }

      const int interval_start = interval_starts[interval_index];
      const int interval_length = interval_lengths[interval_index];
      const int* cur_geom_feats = geom_feats + interval_start * 4;
      const float* cur_x = x + interval_start * c + channel_index;
      float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
                       cur_geom_feats[2] * h * w * c +
                       cur_geom_feats[0] * w * c +
                       cur_geom_feats[1] * c + channel_index;

      float sum = 0.f;
      for (int i = 0; i < interval_length; ++i) {
        sum += cur_x[i * c];
      }
      *cur_out = sum;
    });
  });
}

void bev_pool_grad_xpu(int b, int d, int h, int w, int n, int c,
                       int n_intervals, const float* out_grad,
                       const int* geom_feats, const int* interval_starts,
                       const int* interval_lengths, float* x_grad) {
  (void)b;
  (void)n;
  if (n_intervals == 0 || c == 0) {
    return;
  }

  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int total_work_items = n_intervals * c;
  auto launch = make_1d_launch(total_work_items);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class BevPoolBackward>(launch, [=](sycl::nd_item<1> item) {
      const int idx = item.get_global_linear_id();
      if (idx >= total_work_items) {
        return;
      }
      const int interval_index = idx / c;
      const int channel_index = idx % c;
      if (interval_index >= n_intervals) {
        return;
      }

      const int interval_start = interval_starts[interval_index];
      const int interval_length = interval_lengths[interval_index];
      const int* cur_geom_feats = geom_feats + interval_start * 4;
      float* cur_x_grad = x_grad + interval_start * c + channel_index;
      const float* cur_out_grad = out_grad + cur_geom_feats[3] * d * h * w * c +
                                  cur_geom_feats[2] * h * w * c +
                                  cur_geom_feats[0] * w * c +
                                  cur_geom_feats[1] * c + channel_index;

      for (int i = 0; i < interval_length; ++i) {
        cur_x_grad[i * c] = *cur_out_grad;
      }
    });
  });
}
