# Command-line overrides remain possible: make CXX=/path/to/g++.
CXX := /opt/homebrew/bin/g++-16

TARGET := mnist_app
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src

CPPFLAGS := -I$(INCLUDE_DIR)
# -O3 enables aggressive compiler optimization, such as inlining and loop
# optimization, to reduce the runtime of the numerical training code.
# -ffast-math permits less strict IEEE floating-point transformations. This
# can improve speed, but may slightly change rounding and numerical results.
CXXFLAGS := -std=c++17 -O3 -ffast-math -march=native -fopenmp -g
# The linker also needs OpenMP so calls made by the parallel training loop
# resolve correctly at the final executable-linking step.
LDFLAGS := -fopenmp

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPFILES := $(OBJECTS:.o=.d)

.PHONY: all clean run download

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPFILES)

run: $(TARGET)
	./$(TARGET)

download:
	bash scripts/download_files.sh

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TARGET).dSYM
