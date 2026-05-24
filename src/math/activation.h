#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace math {

template<std::floating_point F, std::size_t N>
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

}