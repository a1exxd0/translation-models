#include "math/activation.h"
#include "math/matrix.h"
#include <array>
#include <concepts>

namespace transformer {

template <std::floating_point F, std::size_t DFF, std::size_t DModel>
auto feed_forward(const std::array<F, DModel> &in,
                  const std::array<std::array<F, DFF>, DModel> &W1,
                  const std::array<F, DFF> &B1,
                  const std::array<std::array<F, DModel>, DFF> &W2,
                  const std::array<F, DModel> &B2) -> std::array<F, DModel> {
  auto inner = math::matmul(in, W1);
  math::add_inplace(B1, inner);
  math::relu(inner);
  auto outer = math::matmul(inner, W2);
  math::add_inplace(B2, outer);
  return outer;
}

} // namespace transformer