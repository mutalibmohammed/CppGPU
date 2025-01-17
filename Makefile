# Compiler and Flags
CXX := nvc++
CXXFLAGS := -std=c++20 -stdpar=gpu --experimental-stdpar -mcmodel=medium -Iinclude
NVCC := nvcc
NVCCFLAGS := -std=c++20

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -traceback -dwarf -Mchkstk -gpu=lineinfo,debug -DNDEBUG -Minfo=all 
    NVCCFLAGS += -g -G
else
    CXXFLAGS += -Ofast  -march=native -Minfo=stdpar
    NVCCFLAGS += -O3
endif



# Source and Object Files
CPP_SRCS := gauss.cpp
CU_SRCS := gauss.cu
SEN_SRCS := gauss_sender.cpp

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

gauss_sender: $(SEN_SRCS)


# Phony Targets (for convenience)
.PHONY: clean

clean:
	rm -rf $(OUT_DIR)