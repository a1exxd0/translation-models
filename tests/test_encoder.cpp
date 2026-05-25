#include "transformer/encoder.h"
#include <gtest/gtest.h>

TEST(EncoderLayer, ForwardRuns) {
  constexpr std::size_t SeqLen = 8;
  constexpr std::size_t DModel = 16;
  constexpr std::size_t NumHeads = 4;
  constexpr std::size_t DFF = 32;

  transformer::EncoderLayer<float, DModel, NumHeads, DFF> layer{};
  std::array<std::array<float, DModel>, SeqLen> x{};
  auto out = layer.forward(x);
  (void)out;
}

TEST(EncoderLayer, ForwardRunsSingleHead) {
  constexpr std::size_t SeqLen = 4;
  constexpr std::size_t DModel = 8;
  constexpr std::size_t NumHeads = 1;
  constexpr std::size_t DFF = 16;

  transformer::EncoderLayer<float, DModel, NumHeads, DFF> layer{};
  std::array<std::array<float, DModel>, SeqLen> x{};
  auto out = layer.forward(x);
  (void)out;
}

TEST(EncoderLayer, ForwardRunsLongSequence) {
  constexpr std::size_t SeqLen = 64;
  constexpr std::size_t DModel = 32;
  constexpr std::size_t NumHeads = 8;
  constexpr std::size_t DFF = 128;

  transformer::EncoderLayer<float, DModel, NumHeads, DFF> layer{};
  std::array<std::array<float, DModel>, SeqLen> x{};
  auto out = layer.forward(x);
  (void)out;
}

TEST(EncoderLayer, ForwardRunsDouble) {
  constexpr std::size_t SeqLen = 4;
  constexpr std::size_t DModel = 8;
  constexpr std::size_t NumHeads = 2;
  constexpr std::size_t DFF = 16;

  transformer::EncoderLayer<double, DModel, NumHeads, DFF> layer{};
  std::array<std::array<double, DModel>, SeqLen> x{};
  auto out = layer.forward(x);
  (void)out;
}