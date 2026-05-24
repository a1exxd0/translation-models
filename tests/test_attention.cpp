#include <gtest/gtest.h>
#include <transformer/attention.h>

TEST(Attention, SimpleSingle) {
    constexpr auto N = 3;
    constexpr auto KeyDims = 3;
    constexpr auto M = 2; 
    constexpr auto ValueDims = 1;

    std::array<std::array<float, KeyDims>, N> Q{{
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
    }};
    
    std::array<std::array<float, KeyDims>, M> K{{
        {1, 2, 0},
        {0, 1, 0},
    }};

    std::array<std::array<float, ValueDims>, M> V = {{
        {1},
        {0},
    }};

    auto result = transformer::scaled_dot_product_attention(Q, K, V);
    for (const auto &row: result) {
        EXPECT_NEAR(row[0], 0.6404f, 1e-4f);
    }
}