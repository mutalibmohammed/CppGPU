#include <iostream>
#include <ranges>
#include <execution>
#include <numeric>
#include <algorithm>
#include <chrono>

template <typename T, int ny, int nx>
void gauss_seidel(const T p[ny][nx], T pnew[ny][nx]) {
    for (const int &y : std::views::iota(1, ny - 1)) {
        for (const int &x : std::views::iota(1, nx - 1)) {
            pnew[y][x] = 0.25 * (pnew[y-1][x] + pnew[y][x-1] + p[y + 1][x] + p[y][x + 1]);
        }
    }
}


template <typename T, int ny, int nx>
void gauss_seidel_wave(const T p[ny][nx], T pnew[ny][nx])
{
    for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++)
    {
        const int xmin = std::max(1, wavefront - (ny - 1 - 1));
        const int xmax = std::min(wavefront - 1, nx - 1 - 1);

        const auto x_range = std::views::iota(xmin, xmax + 1);
        std::for_each(std::execution::par_unseq, x_range.begin(), x_range.end(), [=](int x)
        {
            const int y = wavefront - x;
            pnew[y][x] = 0.25 * (pnew[y-1][x] + pnew[y][x-1] + p[y + 1][x] + p[y][x + 1]);

        });
    }
}


template <typename T, int ny, int nx>
void gauss_seidel_block_wave(const T p[ny][nx], T pnew[ny][nx]) {
    int blocksize_x = 10;
    int blocksize_y = 10;

    int nbx = nx / blocksize_x;
    int nby = ny / blocksize_y;

    for(int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
        int bxmin = std::max(0, bwavefront - (nby - 1));
        int bxmax = std::min(bwavefront, nbx - 1);

        const auto bx_range = std::views::iota(bxmin, bxmax + 1);
           std::for_each(std::execution::par_unseq, bx_range.begin(), bx_range.end(), [=](int bx) {
                int by = bwavefront - bx;
                 
                 int startx = bx * blocksize_x;
                 int starty = by * blocksize_y;

                 for(int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++) {

                     int xmin = std::max(0, wavefront - (blocksize_y - 1));
                     int xmax = std::min(wavefront, blocksize_x - 1);

                     const auto x_range = std::views::iota(xmin, xmax + 1);
                     std::for_each(std::execution::par, x_range.begin(), x_range.end(), [=](int x)
                                   {
                        int y = wavefront - x;
                        y = starty + y;
                        x = startx + x;
                        if(x != 0 && x != nx-1 && y != 0 && y != ny-1)
                            pnew[y][x] = 0.25 * (pnew[y-1][x] + pnew[y][x-1] + p[y + 1][x] + p[y][x + 1]); });
                 }
           });
    }
}

template <typename T, int nx>
void swap_pointer(T (**ptr1)[nx], T (**ptr2)[nx])
{
    auto temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}

int main()
{
    
    using type = double;

    constexpr const int n = 10000;
    constexpr const int nx = 1024;
    constexpr const int ny = 1024;

    auto p    = new type[ny][nx]();
    auto pnew = new type[ny][nx]();

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

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < n; it++)
    {
        //gauss_seidel<type, ny, nx>(p, pnew);
        //gauss_seidel<type, ny, nx>(p, pnew);
        gauss_seidel_block_wave<type, ny, nx>(p, pnew);
        swap_pointer<type, nx>(&p, &pnew);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    type sum = 0.;
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            sum += p[y][x];
        }
    }

    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    delete[] p;
    delete[] pnew;
    return 0;
}


// # count number of floating point operations and compare 
// # look in the kernel and see much memory is being read.
// # talk about perfect cache model vs real cache model.
// # start writing a paragraph every day.
