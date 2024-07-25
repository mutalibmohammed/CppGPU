# Compiler and Flags
CXX := nvc++
CXXFLAGS := -std=c++23 -stdpar=gpu -mcmodel=medium 
NVCC := nvcc
NVCCFLAGS := -std=c++20

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CXXFLAGS +=  -g -Mbounds -traceback -dwarf -Mchkstk  -gpu=lineinfo,debug  -Minfo=all -cuda
    NVCCFLAGS += -g -G
else
    CXXFLAGS += -O3 -Minfo=all
    NVCCFLAGS += -O3
endif



# Source and Object Files
CPP_SRCS := gauss.cpp
CU_SRCS := gauss.cu

# Output Directory
OUT_DIR := out

# Targets
all: gauss gauss_cu

gauss: $(CPP_SRCS)
#	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $^

gauss_cu: $(CU_SRCS)
#	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OUT_DIR)/$@ $^


# Phony Targets (for convenience)
.PHONY: clean

clean:
	rm -rf $(OUT_DIR)