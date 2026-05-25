#include <gtest/gtest.h>
#include <transformer/feed_forward.h>

TEST(FeedForward, SimpleSingle) {
  constexpr auto DModel = 2;
  constexpr auto DFF = 3;

  std::array<float, DModel> in{1, 2};

  std::array<std::array<float, DFF>, DModel> W1{{
      {1, 0, -1},
      {0, 1, 1},
  }};

  std::array<float, DFF> B1{0, 0, 1};

  std::array<std::array<float, DModel>, DFF> W2{{
      {1, 0},
      {0, 1},
      {1, 1},
  }};

  std::array<float, DModel> B2{0, -1};

  auto result = transformer::feed_forward(in, W1, B1, W2, B2);

  EXPECT_NEAR(result[0], 3.0f, 1e-5f);
  EXPECT_NEAR(result[1], 3.0f, 1e-5f);
}

TEST(FeedForward, ReluZerosNegatives) {
  constexpr auto DModel = 2;
  constexpr auto DFF = 2;

  std::array<float, DModel> in{1, 1};

  std::array<std::array<float, DFF>, DModel> W1{{
      {1, -1},
      {1, -1},
  }};
  std::array<float, DFF> B1{0, 0};

  std::array<std::array<float, DModel>, DFF> W2{{
      {1, 0},
      {0, 1},
  }};
  std::array<float, DModel> B2{0, 0};

  auto result = transformer::feed_forward(in, W1, B1, W2, B2);

  EXPECT_NEAR(result[0], 2.0f, 1e-5f);
  EXPECT_NEAR(result[1], 0.0f, 1e-5f);
}