#include <algorithm>
#include <cassert>
#include <chrono>
#include <execution>
#include <experimental/mdspan>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cartesian_product.hpp"

// Godbolt link (Potential bug)
// https://godbolt.org/z/Ecb3rbP3o

template <typename T>
using grid = std::experimental::mdspan<T, std::experimental::dextents<int, 2>>;

template <typename T>
constexpr std::pair<T, T> wavefront_coordinates(T ny, T nx, T wavefront, uint boundary) {
    int left   = boundary & 1;
    int top    = (boundary >> 1) & 1;
    int right  = (boundary >> 2) & 1;
    int bottom = (boundary >> 3) & 1;

    T xmin = std::max(left, wavefront - ((ny - 1 - bottom)));
    T xmax = std::min(wavefront - top, nx - 1 - right);
    return {xmin, xmax >= xmin ? xmax : xmin - 1};
}

// Used to verify results
template <typename T>
void gauss_seidel(const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);
    std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), 1, [=] {
        for (const auto& y : std::views::iota(1, ny - 1)) {
            for (const auto& x : std::views::iota(1, nx - 1)) {
                pnew(y, x) = 0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
            }
        }
    });
}

template <typename T>
void gauss_seidel_wave(const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);
    for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++) {
        const auto [xmin, xmax] = wavefront_coordinates(ny, nx, wavefront, 0b1111);

        const auto x_range = std::views::iota(xmin, xmax + 1);
        std::for_each_n(std::execution::par_unseq, x_range.begin(), x_range.size(), [=](auto x) {
            const auto y = wavefront - x;
            pnew(y, x)   = 0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
        });
    }
}

template <typename T>
void gauss_seidel_wave2(const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);
    for (int wavefront = 2; wavefront < ny + nx - 1; wavefront++) {
        std::for_each_n(
            std::execution::par_unseq, std::views::iota(0).begin(), ny * nx, [=](auto i) {
                const auto [xmin, xmax] = wavefront_coordinates(ny, nx, wavefront, 0b1111);

                // Convert 1D idx to 2D idx
                int x = i & (nx - 1);
                int y = i / nx;

                if (xmin <= x && x <= xmax && wavefront - x == y) {
                    pnew(y, x) =
                        0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
                }
            });
    }
}

template <typename T>
void gauss_seidel_block_wave(const int blocksize_y, const int blocksize_x, const int nby,
                             const int nbx, const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);

    for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
        const auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);

        const auto bx_range = std::views::iota(bxmin, bxmax + 1);

        std::for_each(std::execution::par_unseq, bx_range.begin(), bx_range.end(), [=](auto bx) {
            int by = bwavefront - bx;

            int startx = bx * blocksize_x;
            int starty = by * blocksize_y;

            const uint boundary =
                (by == (nby - 1)) << 3 | (bx == (nbx - 1)) << 2 | (by == 0) << 1 | (bx == 0);

            for (int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++) {
                const auto [xmin, xmax] =
                    wavefront_coordinates(blocksize_y, blocksize_x, wavefront, boundary);

                std::ranges::for_each(std::views::iota(xmin, xmax + 1), [=](auto x) {
                    int y = wavefront - x;
                    y     = starty + y;
                    x     = startx + x;
                    pnew(y, x) =
                        0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
                });
            }
        });
    }
}

template <typename T>
void gauss_seidel_block_wave_2(const int blocksize_y, const int blocksize_x, const int nby,
                               const int nbx, const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);
    // No. of blocks in the longest wavefront
    const int  max_wavefront = std::min(blocksize_x, blocksize_y);
    const auto x_range       = std::views::iota(0, max_wavefront);

    for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
        const auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);
        const auto bx_range       = std::views::iota(bxmin, bxmax + 1);
        auto       idxs           = std::views::cartesian_product(bx_range, x_range);

        for (int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++) {
            std::for_each(std::execution::par_unseq, idxs.begin(), idxs.end(), [=](auto pair) {
                const auto [bx, x] = pair;
                const auto by      = bwavefront - bx;
                const uint boundary =
                    (by == (nby - 1)) << 3 | (bx == (nbx - 1)) << 2 | (by == 0) << 1 | (bx == 0);
                const auto [xmin, xmax] =
                    wavefront_coordinates(blocksize_y, blocksize_x, wavefront, boundary);

                if (x >= xmin && x <= xmax) {
                    const int y = wavefront - x;
                    x           = bx * blocksize_x + x;
                    y           = by * blocksize_y + y;
                    pnew(y, x) =
                        0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
                }
            });
        }
    }
}

template <typename T>
void initialization(grid<T> p) {
    std::fill_n(std::execution::par_unseq, p.data_handle(), p.size(), 0.);
    std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), p.extent(0),
                    [=](auto y) {
                        p(y, 0)               = 10;
                        p(y, p.extent(1) - 1) = 10;
                    });

    std::for_each_n(std::execution::par_unseq, std::views::iota(0).begin(), p.extent(1),
                    [=](auto x) {
                        p(0, x)               = 10;
                        p(p.extent(1) - 1, x) = 10;
                    });
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations> [blocksize_y] [blocksize_x]"
                  << std::endl;
        return 1;
    }
    const int ny          = std::stoi(argv[1]);
    const int nx          = std::stoi(argv[2]);
    const int n           = std::stoi(argv[3]);
    const int blocksize_y = (argc == 6) ? std::stoi(argv[4]) : 16;
    const int blocksize_x = (argc == 6) ? std::stoi(argv[5]) : 16;
    const int nbx         = nx / blocksize_x;
    const int nby         = ny / blocksize_y;

    using type = float;

    assert(ny % blocksize_y == 0 && "ny must be divisible by blocksize");
    assert(nx % blocksize_x == 0 && "nx must be divisible by blocksize");

    std::vector<type> p_data(ny * nx), pnew_data(ny * nx);

    grid<type> p(p_data.data(), ny, nx);
    grid<type> pnew(pnew_data.data(), ny, nx);

    initialization(p);
    initialization(pnew);

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n; ++it) {
#ifdef SERIAL
        gauss_seidel<type>(p, pnew);
#elif defined WAVE
        gauss_seidel_wave<type>(p, pnew);
#elif defined WAVE2
        gauss_seidel_wave2<type>(p, pnew);
#elif defined BLOCK_WAVE
        gauss_seidel_block_wave(blocksize_y, blocksize_x, nby, nbx, p, pnew);
#elif defined BLOCK_WAVE2
        gauss_seidel_block_wave_2<type>(blocksize_y, blocksize_x, nby, nbx, p, pnew);
#else
#error MISSING MACRO DEFINTION TO CHOOSE WAVEFRONT TYPE(SERIAL, WAVE, WAVE2, BLOCK_WAVE, BLOCK_WAVE2)
#endif
        std::swap(p, pnew);
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "Total time: " << duration << " ms\n";

    std::cout << "Memory bandwidth (No Cache Model): "
              << ((ny - 1ull) * (nx - 1ull) * sizeof(type) * 5ull * n) / duration * 1e-6
              << " GB/s\n";
    std::cout << "Memory bandwidth (Perfect Cache Model): "
              << ((ny - 1ull) * (nx - 1ull) * sizeof(type) * 3ull * n) / duration * 1e-6
              << " GB/s\n";

    std::cout << "Compute Throughput: " << ((ny - 1ull) * (nx - 1ull) * 4ull * n) / duration * 1e-6
              << " GFLOPS Precision: " << sizeof(type) << "bytes\n";

    std::cout << ny << " " << nx << " " << n << " " << blocksize_y << " " << blocksize_x
              << std::endl;

    return 0;
}

// # count number of floating point operations and compare
// # look in the kernel and see much memory is being read.
// # talk about perfect cache model vs real cache model.
// # start writing a paragraph every day.
