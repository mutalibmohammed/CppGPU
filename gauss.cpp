#include <iostream>
#include <ranges>
#include <execution>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cassert>
#include <experimental/mdspan>

template <typename T>
using grid_t = std::experimental::mdspan<T, std::experimental::dextents<unsigned long, 2>>;

template <typename T, int ny, int nx>
void gauss_seidel(const T p[ny][nx], T pnew[ny][nx])
{
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

template <typename T, unsigned blocksize_y, unsigned blocksize_x>
void gauss_seidel_block_wave(const grid_t<T> p, grid_t<T> pnew, size_t ny, size_t nx)
{

    const unsigned nbx = nx / blocksize_x;
    const unsigned nby = ny / blocksize_y;

    for (unsigned bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++)
    {
        unsigned bxmin = std::max(0U, bwavefront - (nby - 1));
        unsigned bxmax = std::min(bwavefront, nbx - 1);

        const auto bx_range = std::views::iota(bxmin, bxmax + 1);
        std::for_each_n(std::execution::par_unseq, bx_range.begin(), bx_range.size(), [=](auto bx)
                        {
                unsigned by = bwavefront - bx;

                unsigned startx = bx * blocksize_x;
                unsigned starty = by * blocksize_y;

                for (unsigned wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++)
                {

                    unsigned xmin = std::max(0U, wavefront - (blocksize_y - 1));
                    unsigned xmax = std::min(wavefront, blocksize_x - 1);

                    const auto x_range = std::views::iota(xmin, xmax + 1);
                    std::for_each_n(std::execution::unseq, x_range.begin(), x_range.size(), [=](auto x)
                                    {
                        unsigned y = wavefront - x;
                        y = starty + y;
                        x = startx + x;
                        if(x != 0 && x != nx-1 && y != 0 && y != ny-1)
                            pnew(y, x) = 0.25 * (pnew(y-1, x) + pnew(y, x-1) + p(y + 1, x) + p(y, x + 1)); });
                } });
    }
}

template <typename T, int ny, int nx, int blocksize_y, int blocksize_x>
void gauss_seidel_block_wave_2(const T p[ny][nx], T pnew[ny][nx])
{
        constexpr const int max_wavefront = std::max(blocksize_x, blocksize_y); // No. of blocks in the longest wavefront
        constexpr const int nbx = nx / blocksize_x;
        constexpr const int nby = ny / blocksize_y;

#pragma unroll
    for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++)
    {
        const int bxmin = std::max(0, bwavefront - (nby - 1));
        const int bxmax = std::min(bwavefront, nbx - 1);
        const auto bx_range = std::views::iota(bxmin, bxmax + 1);
        constexpr const auto x_range = std::views::iota(0, max_wavefront);
        
        std::vector<std::pair<int, int>> v {};
        for (int bx : bx_range)
        {
            for (int x : x_range)
            {
                v.push_back(std::make_pair(bx, x));
            }
        }



#pragma unroll
        for (int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++)
        {
            std::for_each(std::execution::par_unseq, v.begin() , v.end(), [=](auto pair)
                          {

                const int bx = pair.first;
                const int x = pair.second;
                const int by = bwavefront - bx;

                const int xmin = std::max(0, wavefront - (blocksize_y - 1));
                const int xmax = std::min(wavefront, blocksize_x - 1);

                if (x >= xmin && x <= xmax)
                {
                    const int y = wavefront - x;
                    const int startx = bx * blocksize_x;
                    const int starty = by * blocksize_y;
                    const int x_ = startx + x;
                    const int y_ = starty + y;
                    if (x_ > 0 && x_ < nx - 1 && y_ > 0 && y_ < ny - 1)
                        pnew[y_][x_] = 0.25 * (pnew[y_ - 1][x_] + pnew[y_][x_ - 1] + p[y_ + 1][x_] + p[y_][x_ + 1]);
                } });
        }
    }
}

// template <typename T, int nx>
// void swap_pointer(T (**ptr1)[nx], T (**ptr2)[nx])
// {
//     auto temp = *ptr1;
//     *ptr1 = *ptr2;
//     *ptr2 = temp;
// }

template <typename T>
void initialization(std::experimental::mdspan<T, std::experimental::dextents<unsigned long, 2>> p)
{
    std::fill_n(std::execution::par_unseq, p.data_handle(), p.size(), 0.);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations>" << std::endl;
        std::terminate();
    }

    using type = double;

    const unsigned long ny = std::stoul(argv[1]);
    const unsigned long nx = std::stoul(argv[2]);
    const unsigned long n = std::stoul(argv[3]);
    constexpr const unsigned blocksize_y = 32;
    constexpr const unsigned blocksize_x = 32;

    assert((ny % blocksize_y == 0, "ny must be divisible by blocksize"));
    assert((nx % blocksize_x == 0, "nx must be divisible by blocksize"));

    std::vector<type> p_data(ny * nx), pnew_data(ny * nx);

    std::experimental::mdspan p{p_data.data(), ny, nx};
    std::experimental::mdspan pnew{pnew_data.data(), ny, nx};

    // auto p = new type[ny][nx];
    // auto pnew = new type[ny][nx];

    initialization<type>(p);
    initialization<type>(pnew);

    // Dirichlet boundary conditions
    std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), ny, [=](auto y)
                    { 
        p(y, 0) = 10;
        pnew(y, 0) = 10;
        p(y, nx - 1) = 10;
        pnew(y, nx - 1) = 10; });

    std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), nx, [=](auto x)
                    {  
        p(0, x) = 10;
        pnew(0, x) = 10;
        p(ny - 1, x) = 10;
        pnew(ny - 1, x) = 10; });

    // for (int x = 0; x < nx; x++)
    // {
    //     p[0][x] = 10;
    //     pnew[0][x] = 10;
    //     p[ny - 1][x] = 10;
    //     pnew[ny - 1][x] = 10;
    // }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < n; it++)
    {
        // gauss_seidel<type, ny, nx>(p, pnew);
        gauss_seidel_block_wave<type, blocksize_y, blocksize_x>(p, pnew, ny, nx);
        // gauss_seidel_block_wave_2<type, ny, nx, blocksize_y, blocksize_x>(p, pnew);
        std::swap(p, pnew);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // type sum = 0.;
    // for (int y = 0; y < ny; y++)
    // {
    //     for (int x = 0; x < nx; x++)
    //     {
    //         sum += p[y][x];
    //     }
    // }

    //std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // delete[] p;
    // delete[] pnew;
    return 0;
}

// # count number of floating point operations and compare
// # look in the kernel and see much memory is being read.
// # talk about perfect cache model vs real cache model.
// # start writing a paragraph every day.
