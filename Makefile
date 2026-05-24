CXX        := g++
CXXSTD     := -std=c++23
WARN       := -Wall -Wextra -Wpedantic
OPT        := -O3 -g

GTEST_DIR  := third_party/googletest/googletest
GTEST_INC  := -I$(GTEST_DIR)/include -I$(GTEST_DIR)
GTEST_SRC  := $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc

SRC_DIR    := src
TEST_DIR   := tests
BUILD_DIR  := build

INCLUDES   := -I$(SRC_DIR) $(GTEST_INC)
CXXFLAGS   := $(CXXSTD) $(WARN) $(OPT) $(INCLUDES) -pthread
LDFLAGS    := -pthread

TEST_SRCS  := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS  := $(TEST_SRCS:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

GTEST_OBJS := $(BUILD_DIR)/gtest-all.o $(BUILD_DIR)/gtest_main.o

# Your library sources, if any
LIB_SRCS   := $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS   := $(LIB_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

TEST_BIN   := $(BUILD_DIR)/run_tests

.PHONY: all test clean
all: $(TEST_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS) $(LIB_OBJS) $(GTEST_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)