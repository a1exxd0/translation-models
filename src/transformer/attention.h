#pragma once

#include "math/activation.h"
#include "math/matrix.h"
#include <array>
#include <concepts>

namespace transformer {

template <std::floating_point F, std::size_t N, std::size_t M,
          std::size_t KeyDims, std::size_t ValueDims>
auto scaled_dot_product_attention_into(
    const std::array<std::array<F, KeyDims>, N> &Q,
    const std::array<std::array<F, KeyDims>, M> &K,
    const std::array<std::array<F, ValueDims>, M> &V, F *out,
    std::size_t out_stride) -> void {
  const auto KT = math::transpose(K);
  auto mult = math::matmul(Q, KT);
  const F divisor = std::sqrt(static_cast<F>(KeyDims));
  for (auto &row : mult) {
    for (auto &elem : row) {
      elem /= divisor;
    }
    math::softmax(row);
  }

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < ValueDims; j++) {
      F acc = F{0};
      for (std::size_t k = 0; k < M; k++) {
        acc += mult[i][k] * V[k][j];
      }
      out[i * out_stride + j] = acc;
    }
  }
}

template <std::floating_point F, std::size_t N, std::size_t M,
          std::size_t KeyDims, std::size_t ValueDims>
auto scaled_dot_product_attention(
    const std::array<std::array<F, KeyDims>, N> &Q,
    const std::array<std::array<F, KeyDims>, M> &K,
    const std::array<std::array<F, ValueDims>, M> &V)
    -> std::array<std::array<F, ValueDims>, N> {
  std::array<std::array<F, ValueDims>, N> result{};
  scaled_dot_product_attention_into(Q, K, V, &result[0][0], ValueDims);
  return result;
}

} // namespace transformer