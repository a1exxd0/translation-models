#include <array>
#include <concepts>
#include "math/matrix.h"
#include "math/activation.h"

namespace transformer {

template<std::floating_point F, std::size_t N, std::size_t M, std::size_t KeyDims, std::size_t ValueDims>
auto scaled_dot_product_attention(
    const std::array<std::array<F, KeyDims>, N> Q,
    const std::array<std::array<F, KeyDims>, M> K,
    const std::array<std::array<F, ValueDims>, M> V) -> std::array<std::array<F, ValueDims>, N> 
{
    constexpr auto KT = math::transpose(K);
    auto mult = math::matmul(Q, KT);
    constexpr F divisor = std::sqrt(static_cast<F>(KeyDims));
    for (auto &row: mult) {
        for (auto &elem: row) {
            elem /= divisor;
        }

        math::softmax(row);
    }

    return math::matmul(mult, V);
}

}