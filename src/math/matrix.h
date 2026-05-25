#pragma once

#include <array>
#include <concepts>

namespace math {

template <std::floating_point F>
constexpr auto matmul_into(const F *a, std::size_t a_stride, const F *b,
                           std::size_t b_stride, F *c, std::size_t c_stride,
                           std::size_t A, std::size_t B, std::size_t C)
    -> void {
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      const F aij = a[i * a_stride + j];
      for (std::size_t k = 0; k < C; k++) {
        c[i * c_stride + k] += aij * b[j * b_stride + k];
      }
    }
  }
}

template <std::floating_point F>
constexpr auto matmul_into(const F *a, const F *b, std::size_t b_stride, F *c,
                           std::size_t c_stride, std::size_t B, std::size_t C)
    -> void {
  for (std::size_t j = 0; j < B; j++) {
    const F aj = a[j];
    for (std::size_t k = 0; k < C; k++) {
      c[k] += aj * b[j * b_stride + k];
    }
  }
  (void)c_stride;
}

template <std::floating_point F, std::size_t A, std::size_t B, std::size_t C>
constexpr auto matmul(const std::array<std::array<F, B>, A> &a,
                      const std::array<std::array<F, C>, B> &b)
    -> std::array<std::array<F, C>, A> {
  std::array<std::array<F, C>, A> c{};
  matmul_into(&a[0][0], B, &b[0][0], C, &c[0][0], C, A, B, C);
  return c;
}

template <std::floating_point F, std::size_t B, std::size_t C>
constexpr auto matmul(const std::array<F, B> &a,
                      const std::array<std::array<F, C>, B> &b)
    -> std::array<F, C> {
  std::array<F, C> c{};
  matmul_into(a.data(), &b[0][0], C, c.data(), C, B, C);
  return c;
}

template <std::floating_point F, std::size_t A, std::size_t B>
constexpr auto matmul_inplace(std::array<std::array<F, B>, A> &a,
                              const std::array<std::array<F, B>, B> &b)
    -> void {
  for (std::size_t i = 0; i < A; i++) {
    std::array<F, B> row{};
    matmul_into(&a[i][0], B, &b[0][0], B, row.data(), B, 1, B, B);
    a[i] = row;
  }
}

template <std::floating_point F, std::size_t N>
constexpr auto matmul_inplace(std::array<F, N> &a,
                              const std::array<std::array<F, N>, N> &b)
    -> void {
  std::array<F, N> row{};
  matmul_into(a.data(), &b[0][0], N, row.data(), N, N, N);
  a = row;
}

template <std::floating_point F, std::size_t N>
constexpr auto add(const std::array<F, N> &a, const std::array<F, N> &b)
    -> std::array<F, N> {
  auto c = std::array<F, N>();
  for (std::size_t i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }

  return c;
}

template <std::floating_point F, std::size_t N, std::size_t M>
constexpr auto add(const std::array<std::array<F, M>, N> &a,
                   const std::array<std::array<F, M>, N> &b)
    -> std::array<std::array<F, M>, N> {
  auto c = std::array<std::array<F, M>, N>();
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }

  return c;
}

template <std::floating_point F, std::size_t N>
constexpr auto add_inplace(std::array<F, N> &a, const std::array<F, N> &b)
    -> void {
  for (std::size_t i = 0; i < N; i++) {
    a[i] = a[i] + b[i];
  }
}

template <std::floating_point F, std::size_t N, std::size_t M>
constexpr auto add_inplace(std::array<std::array<F, M>, N> &a,
                           const std::array<std::array<F, M>, N> &b) -> void {
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      a[i][j] = a[i][j] + b[i][j];
    }
  }
}

template <typename T, std::size_t R, std::size_t C>
constexpr auto transpose(const std::array<std::array<T, C>, R> &in)
    -> std::array<std::array<T, R>, C> {
  std::array<std::array<T, R>, C> out{};
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      out[j][i] = in[i][j];
  return out;
}

} // namespace math