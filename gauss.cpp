#include <iostream>
#include <ranges>
#include <execution>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cassert>
#include <experimental/mdspan>

// Godbolt link (Potential bug)
// https://godbolt.org/z/Ecb3rbP3o

template <typename T>
using grid = std::experimental::mdspan<T, std::experimental::dextents<unsigned long, 2>>;

template <typename T>
void gauss_seidel(const grid<T> p, grid<T> pnew)
{
    const size_t ny = p.extent(0), nx = p.extent(1);
    for (const auto &y : std::views::iota(1ul, ny - 1))
    {
        for (const auto &x : std::views::iota(1ul, nx - 1))
        {
            pnew(y, x) = 0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
        }
    }
}

template <typename T>
void gauss_seidel_wave(const grid<T> p, grid<T> pnew)
{
    const size_t ny = p.extent(0), nx = p.extent(1);
    for (size_t wavefront = 2; wavefront < ny + nx - 1; wavefront++)
    {
        const unsigned xmin = std::max(1ul, wavefront - (ny - 1 - 1));
        const unsigned xmax = std::min(wavefront - 1, nx - 1 - 1);

        const auto x_range = std::views::iota(xmin, xmax + 1);
        std::for_each_n(std::execution::par_unseq, x_range.begin(), x_range.size(), [=](auto x)
                          {
                            const auto y = wavefront - x;
                            pnew(y, x) = 0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1)); });
    }
}

template <typename T, unsigned blocksize_y, unsigned blocksize_x>
void gauss_seidel_block_wave(const grid<T> p, grid<T> pnew)
{
    const size_t ny = p.extent(0), nx = p.extent(1);
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

                    std::ranges::for_each(std::views::iota(xmin, xmax + 1), [=](auto x)
                                          {
                        unsigned y = wavefront - x;
                        y = starty + y;
                        x = startx + x;
                        if(x != 0 && x != nx-1 && y != 0 && y != ny-1)
                            pnew(y, x) = 0.25 * (pnew(y-1, x) + pnew(y, x-1) + p(y + 1, x) + p(y, x + 1)); });
                } });
    }
}

template <typename T, size_t blocksize_y, size_t blocksize_x>
void gauss_seidel_block_wave_2(const grid<T> p, grid<T> pnew)
{
    const size_t ny = p.extent(0), nx = p.extent(1);
    constexpr const size_t max_wavefront = std::max(blocksize_x, blocksize_y); // No. of blocks in the longest wavefront
    const size_t nbx = nx / blocksize_x;
    const size_t nby = ny / blocksize_y;

#pragma unroll
    for (size_t bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++)
    {
        const size_t bxmin = std::max(0ul, bwavefront - (nby - 1));
        const size_t bxmax = std::min(bwavefront, nbx - 1);
        const auto bx_range = std::views::iota(bxmin, bxmax + 1);
        constexpr const auto x_range = std::views::iota(0ul, max_wavefront);

        // TODO: use cartesian product
        std::vector<std::pair<size_t, size_t>> v{};
        for (auto bx : bx_range)
        {
            for (auto x : x_range)
            {
                v.push_back(std::make_pair(bx, x));
            }
        }



#pragma unroll
        for (size_t wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++)
        {
            std::for_each(std::execution::par_unseq, v.begin(), v.end(), [=](auto pair)
                          {

                const auto bx = pair.first;
                const auto x = pair.second;
                const auto by = bwavefront - bx;

                const size_t xmin = std::max(0ul, wavefront - (blocksize_y - 1));
                const size_t xmax = std::min(wavefront, blocksize_x - 1);

                if (x >= xmin && x <= xmax)
                {
                    const size_t y = wavefront - x;
                    const size_t startx = bx * blocksize_x;
                    const size_t starty = by * blocksize_y;
                    const size_t x_ = startx + x;
                    const size_t y_ = starty + y;
                    // if (x_ > 0 && x_ < nx - 1 && y_ > 0 && y_ < ny - 1)
                    pnew(y_, x_) = 0.25 * (pnew(y_ - 1, x_) + pnew(y_, x_ - 1) + p(y_ + 1, x_) + p(y_, x_ + 1));
                } });
        }
    }
}

template <typename T>
void initialization(grid<T> p)
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

    const size_t ny = std::stoul(argv[1]);
    const size_t nx = std::stoul(argv[2]);
    const size_t n = std::stoul(argv[3]);
    constexpr const size_t blocksize_y = 32;
    constexpr const size_t blocksize_x = 32;

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
    // std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), ny, [=](auto y)
    //                 {
    //     p(y, 0) = 10;
    //     pnew(y, 0) = 10;
    //     p(y, nx - 1) = 10;
    //     pnew(y, nx - 1) = 10; });

    // std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), nx, [=](auto x)
    //                 {
    //     p(0, x) = 10;
    //     pnew(0, x) = 10;
    //     p(ny - 1, x) = 10;
    //     pnew(ny - 1, x) = 10; });

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
        // gauss_seidel<type>(p, pnew);
        gauss_seidel_wave<type>(p, pnew);
        // gauss_seidel_block_wave<type, blocksize_y, blocksize_x>(p, pnew);
        // gauss_seidel_block_wave_2<type, blocksize_y, blocksize_x>(p, pnew);
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
