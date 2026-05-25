#pragma once

#include "math/activation.h"
#include "transformer/attention.h"
#include "transformer/feed_forward.h"
#include <array>
#include <concepts>

namespace transformer {

template <std::floating_point F, std::size_t DModel, std::size_t NumHeads,
          std::size_t DFF>
class EncoderLayer {
  static_assert(DModel % NumHeads == 0, "DModel must be divisible by NumHeads");

public:
  static constexpr std::size_t DK = DModel / NumHeads;

private:
  struct HeadWeights {
    std::array<std::array<F, DK>, DModel> WQ;
    std::array<std::array<F, DK>, DModel> WK;
    std::array<std::array<F, DK>, DModel> WV;
  };

  template <std::size_t SeqLen> struct HeadInputs {
    std::array<std::array<F, DK>, SeqLen> Q;
    std::array<std::array<F, DK>, SeqLen> K;
    std::array<std::array<F, DK>, SeqLen> V;
  };

  std::array<HeadWeights, NumHeads> head_weights;
  std::array<std::array<F, DModel>, DModel> output_projection;

  // Feed forward
  std::array<std::array<F, DFF>, DModel> W1;
  std::array<F, DFF> B1;
  std::array<std::array<F, DModel>, DFF> W2;
  std::array<F, DModel> B2;

  // RmsNorms
  std::array<F, DModel> gamma_attention;
  std::array<F, DModel> gamma_ff;

public:
  template <std::size_t SeqLen>
  auto forward(const std::array<std::array<F, DModel>, SeqLen> &x)
      -> std::array<std::array<F, DModel>, SeqLen> {
    auto normed = x;
    for (auto &row : normed) {
      math::rms_norm(row, gamma_attention);
    }

    std::array<std::array<F, DModel>, SeqLen> concat_attentions{};

    for (std::size_t i = 0; i < NumHeads; i++) {
      const auto Q = math::matmul(normed, head_weights[i].WQ);
      const auto K = math::matmul(normed, head_weights[i].WK);
      const auto V = math::matmul(normed, head_weights[i].WV);

      transformer::scaled_dot_product_attention_into(
          Q, K, V, &concat_attentions[0][i * DK], DModel);
    }

    math::matmul_inplace(concat_attentions, output_projection);
    math::add_inplace(concat_attentions, x);

    auto normed_ff = concat_attentions;
    for (auto &row : normed_ff) {
      math::rms_norm(row, gamma_ff);
    }

    for (std::size_t i = 0; i < SeqLen; i++) {
      auto ff = feed_forward(normed_ff[i], W1, B1, W2, B2);
      math::add_inplace(concat_attentions[i], ff);
    }

    return concat_attentions;
  }
};

} // namespace transformer