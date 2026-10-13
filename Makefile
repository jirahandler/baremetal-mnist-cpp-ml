# Command-line overrides remain possible: make CXX=/path/to/g++.
CXX := /opt/homebrew/bin/g++-16

TARGET := mnist_app
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src

# Homebrew's header root. All Homebrew-installed libraries put their
# headers here, so one -I covers yaml-cpp, libomp, and anything else
# you add later.
CPPFLAGS := -I$(INCLUDE_DIR) -I/opt/homebrew/include

# -O3 enables aggressive compiler optimization.
# -ffast-math permits reassociation of floating-point operations.
# -fopenmp enables the OpenMP pragmas in neural_network.cpp.
# -g adds debug symbols for gdb.
CXXFLAGS := -std=c++17 -O3 -ffast-math -march=native -fopenmp -g

# -fopenmp on the link line pulls in the OpenMP runtime.
# -L/opt/homebrew/lib tells the linker where to find Homebrew libraries.
LDFLAGS := -fopenmp -L/opt/homebrew/lib

# Libraries to link, listed AFTER the object files on the link line.
# -lyaml-cpp resolves the YAML:: symbols used by config.cpp.
LDLIBS := -lyaml-cpp

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPFILES := $(OBJECTS:.o=.d)

.PHONY: all clean run download

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

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