#include <iostream>
#include <numeric>

#define WAVEFRONT_TYPE

__host__ __device__ constexpr int _max(int a, int b) {
    return a > b ? a : b;
}

__host__ __device__ constexpr int _min(int a, int b) {
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

// void test_wavefront_coordinates() {
//     auto constexpr result = wavefront_coordinates(10, 10, 9, 0b1111);
//     static_assert(result.first == 1);
//     static_assert(result.second == 8, "Second value is not 7");
//     std::cout << result.second << std::endl;
// }

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
__global__ void gauss_seidel_wave(const int ny,const int nx, const T* p, T* pnew) {
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

template <typename T>
__global__ void gauss_seidel_wave(const int ny, const int nx, const int wavefront, const T* p,
                                  T* pnew) {
    auto [xmin, xmax] = wavefront_coordinates(ny, nx, wavefront, 0b1111);

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= xmin && x <= xmax) {
        int y            = wavefront - x;
        pnew[y * nx + x] = 0.25 * (pnew[(y - 1) * nx + x] + pnew[y * nx + (x - 1)] +
                                   p[(y + 1) * nx + x] + p[y * nx + (x + 1)]);
    }
}

template <typename T>
__global__ void gauss_seidel_block_wave(int nby, int nbx, int ny, int nx, const T* p, T* pnew,
                                        int bwavefront) {
    // Given blockid.x and bwavefront, calculate the startx and starty
    // BlockIdx.x represents the Number of the block on thw wavefront. O is the leftmost block on
    // the wavefront.
    int bxmin = max(0, bwavefront - ((nby - 1)));
    int bxmax = min(bwavefront, nbx - 1);

    int bx = blockIdx.x + bxmin;
    int by = bwavefront - bx;

    int blocksize_x = nx / nbx;
    int blocksize_y = ny / nby;

    // printf("B Wavefront: %d, Block(%2d): %d, %d\n", bwavefront, blockIdx bx, by);

    if (bx > bxmax) {
        printf("Block out of bounds\n");
    }

    int startx = bx * blocksize_x;
    int starty = by * blocksize_y;

    for (int wavefront = 0; wavefront < blocksize_y + blocksize_x - 1; wavefront++) {
        int xmin = max(0, wavefront - (blocksize_y - 1));
        int xmax = min(wavefront, blocksize_x - 1);

        int x = threadIdx.x;

        if (x >= xmin && x <= xmax) {
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

int main() {
    typedef double gtype;

    constexpr const int iterations = 3;
    constexpr const int nx         = 1024;
    constexpr const int ny         = 1024;

    auto p    = new gtype[ny][nx]();
    auto pnew = new gtype[ny][nx]();

    gtype* d_p;
    gtype* d_pnew;

    cudaError_t err;

    err = cudaMalloc((void**)&d_p, ny * nx * sizeof(gtype));
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void**)&d_pnew, ny * nx * sizeof(gtype));
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    // Dirichlet boundary conditions
    for (int y = 0; y < ny; y++) {
        p[y][0]         = 10;
        pnew[y][0]      = 10;
        p[y][nx - 1]    = 10;
        pnew[y][nx - 1] = 10;
    }

    for (int x = 0; x < nx; x++) {
        p[0][x]         = 10;
        pnew[0][x]      = 10;
        p[ny - 1][x]    = 10;
        pnew[ny - 1][x] = 10;
    }

    err = cudaMemcpy(d_p, p, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(d_pnew, pnew, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    int blocksize_x = 10;
    int blocksize_y = 10;

    int nbx = nx / blocksize_x;
    int nby = ny / blocksize_y;

    dim3 blocksize(16, 16);
    dim3 gridsize(1024 / 16, 1024 / 16);



    for (int i = 0; i < iterations; i++) {

#if SERIAL
            gauss_seidel<<<1, 1>>>(ny, nx, d_p, d_pnew);
#elif WAVE 
            // todo fix blocksize
            gauss_seidel_wave<<<1, nx>>>(ny, nx, d_p, d_pnew);

#elif MULTI_BLOCK_WAVE
            gauss_seidel_

#elif WAVEFRONT_TYPE == 3

#endif
            gauss_seidel_wave<<<gridsize, blocksize>>>(ny, nx, wavefront, d_p, d_pnew);

            // gauss_seidel_wave<<<1, 1>>>(ny, nx, d_p, d_pnew);
            // cudaError_t errSync = cudaGetLastError();
            // if (errSync != cudaSuccess)
            //     printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));

            // cudaError_t errAsync = cudaDeviceSynchronize();
            // if (errAsync != cudaSuccess)
            //     printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

            //     gauss_seidel<<<1, std::max(nx, ny)>>>(ny, nx, (i % 2 == 0) ? d_p : d_pnew,
            //                                           (i % 2 == 0) ? d_pnew : d_p);
        

        // cudaError_t errSync = cudaGetLastError();
        // cudaError_t errAsync = cudaDeviceSynchronize();
        // if (errSync != cudaSuccess)
        //     printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        // if (errAsync != cudaSuccess)
        //     printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

        // for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++)
        // {

        //     // Figure out the number of blocks on the wavefront
        //     auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);
        //     // int bxmin = max(0, bwavefront - (nby - 1));
        //     // int bxmax = min(bwavefront, nbx - 1);
        //     int num_blocks = bxmax - bxmin + 1;

        //     std::cout << "Wavefront: " << bwavefront << " Blocks: " << num_blocks <<
        //     std::endl;

        //     // Call the kernel with the number of blocks
        //     gauss_seidel_block_wave<<<num_blocks, std::min(blocksize_y, blocksize_x)>>>(nby,
        //     nbx, ny, nx, (i % 2 == 0) ? d_p : d_pnew, (i % 2 == 0) ? d_pnew : d_p,
        //     bwavefront);

        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess)
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        // }
    }

    err = cudaMemcpy(p, d_p, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(pnew, d_pnew, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    gtype sum = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            sum += p[y][x];
        }
    }
    std::cout << "Sum: " << sum << std::endl;
    delete[] p;
    delete[] pnew;

    cudaFree(d_p);
    cudaFree(d_pnew);
    return 0;
}