# Compiler and Flags
CXX := nvc++
CXXFLAGS := -std=c++20 -stdpar=gpu  -mcmodel=medium -Iinclude -Istdexec/include/  -DWAVE2 -Minfo=stdpar -Wpedantic
NVCC := nvcc
NVCCFLAGS := -std=c++20 --expt-relaxed-constexpr -lineinfo -Xcompiler -Wall  -DWAVE2 

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -traceback -dwarf -Mchkstk -gpu=lineinfo,debug
    NVCCFLAGS += -g -DDEBUG
else
    CXXFLAGS +=  -O4 -fast -mtune=native  -gpu=lineinfo,fastmath -fstrict-aliasing -Msafeptr -Minline -DNDEBUG 
    NVCCFLAGS += -O4 -arch=native -lto -DNDEBUG
endif



# Source and Object Files
CPP_SRCS := gauss.cpp
CU_SRCS := gauss.cu
SEN_SRCS := gauss_sender.cpp

# Output Directory
OUT_DIR := out

# Targets
all: gauss gauss_cu gauss_sender

gauss_sender: $(SEN_SRCS)
	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $^

gauss: $(CPP_SRCS)
	mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $^

gauss_cu: $(CU_SRCS)
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OUT_DIR)/$@ $^


  

# Phony Targets (for convenience)
.PHONY: clean

clean:
	rm -rf $(OUT_DIR)
