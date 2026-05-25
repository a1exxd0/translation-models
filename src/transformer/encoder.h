#include "transformer/attention.h"
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
};

} // namespace transformer