#include <gtest/gtest.h>
#include "math/matrix.h" 

TEST(Transpose, SquareMatrix) {
    std::array<std::array<float, 3>, 3> in{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};
    auto out = math::transpose(in);
    EXPECT_EQ(out[0][1], 4.0f);
    EXPECT_EQ(out[2][0], 3.0f);
}

TEST(Transpose, NonSquareMatrix) {
    std::array<std::array<float, 3>, 3> in{{
        {1, 2, 3},
        {4, 5, 6},
    }};
    auto out = math::transpose(in);
    EXPECT_EQ(out[0][1], 4.0f);
    EXPECT_EQ(out[2][1], 6.0f);
}

TEST(Matmul, SquareMatrix) {
    std::array<std::array<float, 3>, 3> in1{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};
    std::array<std::array<float, 3>, 3> in2{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};

    auto out = math::matmul(in1, in2);
    EXPECT_EQ(out[1][1], 81.0f);
}

TEST(Matmul, NonSquareMatrix) {
    std::array<std::array<float, 3>, 3> in1{{
        {1, 2, 3},
        {4, 5, 6},
    }};
    std::array<std::array<float, 3>, 3> in2{{
        {1, 2},
        {4, 5},
        {7, 8},
    }};

    auto out = math::matmul(in1, in2);
    EXPECT_EQ(out[0][1], 36.0f);
}