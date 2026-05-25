#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <numeric>

namespace math {

template <std::floating_point F, std::size_t N>
auto softmax(std::array<F, N> &a) -> void {
  static_assert(N > 0);

  F m = *std::ranges::max_element(a);
  F sum = 0;
  for (std::size_t i = 0; i < N; i++) {
    a[i] = std::exp(a[i] - m);
    sum += a[i];
  }

  const F inv = F{1} / sum;
  for (std::size_t i = 0; i < N; i++) {
    a[i] *= inv;
  }
}

template <std::floating_point F, std::size_t N>
auto rms_norm(std::array<F, N> &a, const std::array<F, N> &gamma) -> void {
  const F inv_n = F{1} / static_cast<F>(N);
  const F ms = inv_n * std::accumulate(a.begin(), a.end(), F{0},
                                       [](F acc, F v) { return acc + v * v; });
  const F inv_rms = F{1} / std::sqrt(ms + F{1e-5});
  for (std::size_t i = 0; i < N; i++) {
    a[i] = gamma[i] * a[i] * inv_rms;
  }
}

template <std::floating_point F, std::size_t N>
auto relu(std::array<F, N> &a) -> void {
  for (std::size_t i = 0; i < N; i++) {
    a[i] = std::max(static_cast<F>(0), a[i]);
  }
}

} // namespace math