#include <cassert>
#include <iostream>
#include <numeric>

#ifdef DEBUG
// clang-format off
#define CHECK_CUDA(call) do { cudaError_t status = call; if (status != cudaSuccess) { std::fprintf(stderr, "CUDA Error at line %d in %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(status));} } while (0)
// clang-format on
#else
#define CHECK_CUDA(call) \
    do {                 \
        call;            \
    } while (0)
#endif

__host__ __device__ constexpr inline int _max(int a, int b) {
    return a > b ? a : b;
}

__host__ __device__ constexpr inline int _min(int a, int b) {
    return a < b ? a : b;
}

__host__ __device__ constexpr std::pair<int, int> wavefront_coordinates(int ny, int nx,
                                                                        int  wavefront,
                                                                        uint boundary) {
    int left   = boundary & 1;
    int top    = (boundary >> 1) & 1;
    int right  = (boundary >> 2) & 1;
    int bottom = (boundary >> 3) & 1;

    int xmin = _max(left, wavefront - ((ny - 1 - bottom)));
    int xmax = _min(wavefront - top, nx - 1 - right);
    return {xmin, xmax};
}

// Dirichlet boundary conditions
template <typename T>
__global__ void initialize(T* p, T* pnew) {
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = gridDim.x * blockDim.x;
    int ny = gridDim.y * blockDim.y;

    p[y * nx + x]    = 0;
    pnew[y * nx + x] = 0;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        p[y * nx + x]    = 10;
        pnew[y * nx + x] = 10;
    }
}

template <typename T>
__global__ void gauss_seidel(const int ny, const int nx, const T* p, T* pnew) {
    for (int i = 1; i < ny - 1; i++) {
        for (int j = 1; j < nx - 1; j++) {
            pnew[i * nx + j] = 0.25 * (pnew[(i - 1) * nx + j] + pnew[i * nx + (j - 1)] +
                                       p[(i + 1) * nx + j] + p[i * nx + (j + 1)]);
        }
    }
}

template <typename T>
__global__ void gauss_seidel_wave(const int ny, const int nx, const T* p, T* pnew) {
    for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++) {
        auto [xmin, xmax] = wavefront_coordinates(ny, nx, wavefront, 0b1111);

        int x = threadIdx.x;

        if (x >= xmin && x <= xmax) {
            int y            = wavefront - x;
            pnew[y * nx + x] = 0.25 * (pnew[(y - 1) * nx + x] + pnew[y * nx + (x - 1)] +
                                       p[(y + 1) * nx + x] + p[y * nx + (x + 1)]);
        }

        __syncthreads();
    }
}

// Wave 2
template <typename T>
__global__ void gauss_seidel_wave(const int wavefront, const T* p, T* pnew) {
    const int nx = gridDim.x * blockDim.x;
    const int ny = gridDim.y * blockDim.y;

    auto [xmin, xmax] = wavefront_coordinates(ny, nx, wavefront, 0b1111);
    const auto x      = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y      = blockIdx.y * blockDim.y + threadIdx.y;

    if (xmin <= x && x <= xmax && wavefront - x == y) {
        const auto i = y * nx + x;
        pnew[i]      = 0.25 * (pnew[i - nx] + pnew[i - 1] + p[i + nx] + p[i + 1]);
    }
}

template <typename T>
/**
 * @brief Performs the Gauss-Seidel method using block wave approach.
 *
 * Divides the grid into blocks (similar to CUDA programming model). Blocks on a wavefront
 * can be executed in parallel. Kernel is launched with the same no. of blocks as the blocks on the
 * wavefront (in 1D). The no. of threads is equal to the maximum length of a wavefront within a
 * block. Some calculation is required to figure out addresses based on the block id and the
 * bwavefront.
 *
 * @param nby The number of blocks in the y-direction.
 * @param nbx The number of blocks in the x-direction.
 * @param ny The total number of rows in the grid.
 * @param nx The total number of columns in the grid.
 */
__global__ void gauss_seidel_block_wave(const int nby, const int nbx, const int ny, const int nx,
                                        const T* p, T* pnew, const int bwavefront) {
    // Given blockid.x and bwavefront, calculate the startx and starty
    // BlockIdx.x represents the Number of the block on the wavefront. O is the leftmost block on
    // the wavefront.
    int bxmin = max(0, bwavefront - ((nby - 1)));
    int bxmax = min(bwavefront, nbx - 1);

    int bx = blockIdx.x + bxmin;
    int by = bwavefront - bx;

    int blocksize_x = nx / nbx;
    int blocksize_y = ny / nby;

    // printf("B Wavefront: %d, Block(%2d): %d, %d\n", bwavefront, blockIdx bx, by);

    assert(bx <= bxmax && "Block out of bounds");

    int startx = bx * blocksize_x;
    int starty = by * blocksize_y;

    for (int wavefront = 0; wavefront < blocksize_y + blocksize_x - 1; wavefront++) {
        int xmin = max(0, wavefront - (blocksize_y - 1));
        int xmax = min(wavefront, blocksize_x - 1);

        int x = threadIdx.x;

        if (x >= xmin && x <= xmax) {
            // TODO resolve this
            int y = wavefront - x;
            y     = y + starty;
            x     = x + startx;
            if (x != 0 && x != nx - 1 && y != 0 && y != ny - 1)
                pnew[y * nx + x] = 0.25 * (pnew[(y - 1) * nx + x] + pnew[y * nx + (x - 1)] +
                                           p[(y + 1) * nx + x] + p[y * nx + (x + 1)]);
        }

        __syncthreads();
    }
}

template <typename T, uint blksize>
__global__ void error(const T* p, const T* pnew, T* d_error) {
    extern __shared__ T shared_mem[];
    auto                i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    shared_mem[threadIdx.x] =
        max(abs(pnew[i] - p[i]), abs(pnew[i + blockDim.x] - p[i + blockDim.x]));
    __syncthreads();

    for (unsigned stride = blksize / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] =
                max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        if (blksize >= 64)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 32]);
        if (blksize >= 32)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 16]);
        if (blksize >= 16)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 8]);
        if (blksize >= 8)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 4]);
        if (blksize >= 4)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 2]);
        if (blksize >= 2)
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + 1]);
    }

    if (threadIdx.x == 0) {
        d_error[blockIdx.x] = shared_mem[0];
    }
}

template <typename T>
__global__ void reduce(T* d_error) {
    extern __shared__ T shared_mem[];
    shared_mem[threadIdx.x] = d_error[threadIdx.x];
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared_mem[threadIdx.x] =
                max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
    }

    if (threadIdx.x == 0) {
        d_error[0] = shared_mem[0];
    }
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations> [blocksize_y] [blocksize_x]"
                  << std::endl;
        return 1;
    }
    const int              ny              = std::stoi(argv[1]);
    const int              nx              = std::stoi(argv[2]);
    const int              n               = std::stoi(argv[3]);
    const int              blocksize_y     = (argc == 6) ? std::stoi(argv[4]) : 16;
    const int              blocksize_x     = (argc == 6) ? std::stoi(argv[5]) : 16;
    const int              nbx             = nx / blocksize_x;
    const int              nby             = ny / blocksize_y;
    constexpr const size_t reduce_blksize  = 128;
    const size_t           reduce_gridsize = ny * nx / (reduce_blksize * 2);

    assert(ny % blocksize_y == 0 && "ny must be divisible by blocksize");
    assert(nx % blocksize_x == 0 && "nx must be divisible by blocksize");

    typedef float gtype;

    auto p    = new gtype[ny * nx];
    auto pnew = new gtype[ny * nx];
    auto perr = new gtype;

    gtype* d_p;
    gtype* d_pnew;
    gtype* d_error;

    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CHECK_CUDA(cudaMalloc((void**)&d_p, ny * nx * sizeof(gtype)));
    CHECK_CUDA(cudaMalloc((void**)&d_pnew, ny * nx * sizeof(gtype)));
    CHECK_CUDA(cudaMalloc((void**)&d_error, ny * nx * sizeof(gtype)));

    initialize<<<dim3(nbx, nby), dim3(blocksize_x, blocksize_y)>>>(d_p, d_pnew);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < n; i++) {
#ifdef SERIAL
        gauss_seidel<<<1, 1>>>(ny, nx, d_p, d_pnew);
#elif defined WAVE
        gauss_seidel_wave<<<1, std::min(nx, ny)>>>(ny, nx, d_p, d_pnew);

#elif defined WAVE2

        for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++)
            gauss_seidel_wave<<<dim3(nbx, nby), dim3(blocksize_x, blocksize_y)>>>(wavefront, d_p,
                                                                                  d_pnew);

#elif defined BLOCK_WAVE
        for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
            // Figure out the number of blocks on the wavefront
            const auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);
            int num_blocks            = bxmax - bxmin + 1;

            // std::cout << "Wavefront: " << bwavefront << " Blocks: " << num_blocks << std::endl;

            // Call the kernel with the number of blocks
            gauss_seidel_block_wave<<<num_blocks, std::min(blocksize_x, blocksize_y)>>>(
                nby, nbx, ny, nx, d_p, d_pnew, bwavefront);
        }
#else
#error MISSING MACRO DEFINTION TO CHOOSE WAVEFRONT TYPE(SERIAL, WAVE, WAVE2, BLOCK_WAVE)
#endif

#ifdef REDUCE
        error<gtype, reduce_blksize>
            <<<reduce_gridsize, reduce_blksize, reduce_blksize * sizeof(gtype)>>>(d_p, d_pnew,
                                                                                  d_error);
        reduce<<<1, reduce_blksize, reduce_blksize * sizeof(gtype)>>>(d_error);
        CHECK_CUDA(cudaMemcpy(perr, d_error, sizeof(gtype), cudaMemcpyDeviceToHost));
        if (*perr < 1e-6) {
            std::printf("Converged at iteration %d MAE: %f\n", i, *perr);
            break;
        }
#endif

        std::swap(d_p, d_pnew);

#ifdef DEBUG
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
#endif
    }

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaMemcpy(p, d_p, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(pnew, d_pnew, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    unsigned long long bytes = (ny - 1ull) * (nx - 1ull) * n;
    std::printf("Total time: %f ms\n", milliseconds);
    std::printf("Memory bandwidth (No Cache Model): %f GB/s\n",
                (5ull * bytes * sizeof(gtype) / milliseconds * 1e-6));
    std::printf("Memory bandwidth (Perfect Cache Model): %f GB/s\n",
                (3ull * bytes * sizeof(gtype) / milliseconds * 1e-6));
    std::printf("Compute Throughput: %f GFLOPS Precision %lu bytes\n",
                4ull * bytes / milliseconds * 1e-6, sizeof(gtype));
    std::printf("%d %d %d %d %d\n", ny, nx, n, blocksize_x, blocksize_y);

    delete[] p;
    delete[] pnew;
    delete perr;

    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_pnew));
    CHECK_CUDA(cudaFree(d_error));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}