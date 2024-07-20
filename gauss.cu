#include <iostream>
#include <numeric>

template <typename T>
__global__ void gauss_seidel(int ny, int nx, const T *p, T *pnew)
{
    for (int i = 1; i < ny - 1; i++)
    {
        for (int j = 1; j < nx - 1; j++)
        {
            pnew[i * nx + j] = 0.25 * (pnew[(i - 1) * nx + j] + pnew[i * nx + (j - 1)] + p[(i + 1) * nx + j] + p[i * nx + (j + 1)]);
        }
    }
}

template <typename T>
__global__ void gauss_seidel_wave(int ny, int nx, const T *p, T *pnew)
{

    for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++)
    {

        int xmin = max(1, wavefront - ((ny - 1) - 1));
        // TODO: maybe change it so its less than or equal to.
        int xmax = min(wavefront, nx - 1);

        int x = threadIdx.x;

        if (x >= xmin && x < xmax)
        {
            int y = wavefront - x;
            pnew[y * nx + x] = 0.25 * (pnew[(y - 1) * nx + x] + pnew[y * nx + (x - 1)] + p[(y + 1) * nx + x] + p[y * nx + (x + 1)]);
        }

        __syncthreads();
    }
}

template <typename T>
__global__ void gauss_seidel_block_wave(int nby, int nbx, int ny, int nx, const T *p, T *pnew, int bwavefront)
{
    // Given blockid.x and bwavefront, calculate the startx and starty
    // BlockIdx.x represents the Number of the block on thw wavefront. O is the leftmost block on the wavefront.
    int bxmin = max(0, bwavefront - ((ny / nby - 1)));
    int bxmax = min(bwavefront, nx / nbx - 1);

    int bx = blockIdx.x + bxmin;
    int by = bwavefront - bx;

    // printf("B Wavefront: %d, Block(%2d): %d, %d\n", bwavefront, blockIdx bx, by);

    if (bx > bxmax)
    {
        printf("Block out of bounds\n");
    }

    int startx = bx * nbx;
    int starty = by * nby;

    // TODO: Handle block boundaries

    for (int wavefront = 0; wavefront < nby + nbx - 1; wavefront++)
    {

        int xmin = max(1, wavefront - ((nby - 1) - 1));
        int xmax = min(wavefront, nbx - 1);

        int x = threadIdx.x;

        if (x >= xmin && x <= xmax)
        {
            int y = wavefront - x;
            y = y + starty;
            x = x + startx;
            if(x != 0 && x != nx-1 && y != 0 && y != ny-1)
                pnew[y * nx + x] = 0.25 * (pnew[(y - 1) * nx + x] + pnew[y * nx + (x - 1)] + p[(y + 1) * nx + x] + p[y * nx + (x + 1)]);
        }

        __syncthreads();
    }
}

int main()
{

    typedef double gtype;

    constexpr const int iterations = 10000;
    constexpr const int nx = 100;
    constexpr const int ny = 100;

    auto p = new gtype[ny][nx]();
    auto pnew = new gtype[ny][nx]();

    gtype *d_p;
    gtype *d_pnew;

    cudaError_t err;

    err = cudaMalloc((void **)&d_p, ny * nx * sizeof(gtype));
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **)&d_pnew, ny * nx * sizeof(gtype));
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    // Dirichlet boundary conditions
    for (int y = 0; y < ny; y++)
    {
        p[y][0] = 10;
        pnew[y][0] = 10;
        p[y][nx - 1] = 10;
        pnew[y][nx - 1] = 10;
    }

    for (int x = 0; x < nx; x++)
    {
        p[0][x] = 10;
        pnew[0][x] = 10;
        p[ny - 1][x] = 10;
        pnew[ny - 1][x] = 10;
    }

    err = cudaMemcpy(d_p, p, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(d_pnew, pnew, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    int nbx = 20;
    int nby = 20;

    for (int i = 0; i < iterations; i++)
    {
        // gauss_seidel<<<1, std::max(nx, ny)>>>(ny, nx, (i % 2 == 0) ? d_p : d_pnew, (i % 2 == 0) ? d_pnew : d_p);
        // cudaError_t errSync = cudaGetLastError();
        // cudaError_t errAsync = cudaDeviceSynchronize();
        // if (errSync != cudaSuccess)
        //     printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        // if (errAsync != cudaSuccess)
        //     printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

        for(int bwavefront = 0; bwavefront < ny/nby + nx/nbx - 1; bwavefront++) {

            // Figure out the number of blocks on the wavefront
            int bxmin = max(0, bwavefront - ((ny/nby - 1) ));
            int bxmax = min(bwavefront, nx/nbx - 1);
            int num_blocks = bxmax - bxmin + 1;

            // Call the kernel with the number of blocks
            gauss_seidel_block_wave<<<num_blocks, std::min(nbx, nby)>>>(nby, nbx, ny, nx, (i % 2 == 0) ? d_p : d_pnew, (i % 2 == 0) ? d_pnew : d_p, bwavefront);

            cudaError_t errSync = cudaGetLastError();
            cudaError_t errAsync = cudaDeviceSynchronize();
            if (errSync != cudaSuccess)
                printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
            if (errAsync != cudaSuccess)
                printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

            }
    }

    err = cudaMemcpy(p, d_p, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(pnew, d_pnew, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    gtype sum = 0;
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
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