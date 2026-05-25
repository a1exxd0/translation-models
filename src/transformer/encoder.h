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
  std::array<std::tuple<std::array<std::array<F, DK>, DModel>,
                        std::array<std::array<F, DK>, DModel>,
                        std::array<std::array<F, DK>, DModel>>,
             NumHeads>
      head_weights;

  std::array<std::array<F, DModel>, DModel> output_projection;
};

} // namespace transformer