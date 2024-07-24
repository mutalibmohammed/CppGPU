# Compiler and Flags
CXX := nvc++
CXXFLAGS := -std=c++23 -stdpar=gpu -gpu=lineinfo -Minfo=all
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++20

# Source and Object Files
CPP_SRCS := gauss.cpp
CPP_OBJS := $(addprefix out/,$(CPP_SRCS:.cpp=.o))
CU_SRCS := gauss.cu
CU_OBJS := $(addprefix out/,$(CU_SRCS:.cu=_cu.o))

# Output Directory
OUT_DIR := out

# Targets
all: gauss gauss_cu

gauss: $(CPP_OBJS)
#	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $^

gauss_cu: $(CU_OBJS)
#	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OUT_DIR)/$@ $^

out/%.o: %.cpp
#	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

out/%_cu.o: %.cu
#	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Phony Targets (for convenience)
.PHONY: clean

clean:
	rm -rf $(OUT_DIR)