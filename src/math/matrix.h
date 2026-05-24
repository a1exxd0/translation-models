#pragma once

#include <array>

namespace math {

template<std::floating_point F, std::size_t A, std::size_t B, std::size_t C>
auto matmul(
    const std::array<std::array<F, B>, A> &a, 
    const std::array<std::array<F, C>, B> &b) -> std::array<std::array<F, C>, A> 
{
    auto c = std::array<std::array<F, C>, A>{};
    for (std::size_t i = 0; i < A; i++) {
        for (std::size_t j = 0; j < B; j++) {
            const auto aij = a[i][j];
            for (std::size_t k = 0; k < C; k++) {
                c[i][k] += aij * b[j][k];
            }
        }
    }

    return c;
}

template <typename T, std::size_t R, std::size_t C>
constexpr auto transpose(
    const std::array<std::array<T, C>, R>& in) -> std::array<std::array<T, R>, C> 
{
    std::array<std::array<T, R>, C> out{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            out[j][i] = in[i][j];
    return out;
}

}