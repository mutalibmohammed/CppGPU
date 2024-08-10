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

template <typename T, int blocksize_y, int blocksize_x>
void gauss_seidel_block_wave(const grid<T> p, grid<T> pnew) {
    const int ny = p.extent(0), nx = p.extent(1);
    const int nbx = nx / blocksize_x;
    const int nby = ny / blocksize_y;

    // printf("nbx: %d, nby: %d\n", nbx, nby);

    for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
        const auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);

        const auto bx_range = std::views::iota(bxmin, bxmax + 1);
        // printf("bxmin: %d, bxmax: %d, bwavefront: %d, size: %d\n", bxmin, bxmax, bwavefront,
        // bx_range.size());

        std::for_each(std::execution::par_unseq, bx_range.begin(), bx_range.end(), [=](auto bx) {
            // printf("bwavefront: %d\n", bwavefront);
            // printf("===bx: %d\n", bx);
            int by = bwavefront - bx;

            int startx = bx * blocksize_x;
            int starty = by * blocksize_y;

            // printf("bwavefront: %d, bx: %d, by: %d, startx: %d, starty: %d\n", bwavefront, bx,
            // by, startx, starty);

            const uint boundary =
                (by == (nby - 1)) << 3 | (bx == (nbx - 1)) << 2 | (by == 0) << 1 | (bx == 0);

            // printf("=== bwavefront: %d, boundary: %u, bx: %d, by: %d\n", bwavefront, boundary,
            // bx, by);

            for (int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++) {
                const auto [xmin, xmax] =
                    wavefront_coordinates(blocksize_y, blocksize_x, wavefront, boundary);
                std::ranges::for_each(std::views::iota(xmin, xmax + 1), [=](auto x) {
                    int y = wavefront - x;
                    // printf("wavefront: %d, x: %d, y: %d\n", wavefront, x, y);
                    y = starty + y;
                    x = startx + x;

                    // printf("wavefront: %d, x: %d, y: %d\n", wavefront, x, y);
                    // if (x < 1 || x >= nx - 1 || y >= 1 && y < ny - 1)
                    //     pnew(y, x) = 0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y,
                    //     x + 1));
                    pnew(y, x) =
                        0.25 * (pnew(y - 1, x) + pnew(y, x - 1) + p(y + 1, x) + p(y, x + 1));
                });
            }
        });
    }
}

template <typename T, int blocksize_y, int blocksize_x>
void gauss_seidel_block_wave_2(const grid<T> p, grid<T> pnew) {
    const int           ny = p.extent(0), nx = p.extent(1);
    constexpr const int max_wavefront =
        std::max(blocksize_x, blocksize_y);  // No. of blocks in the longest wavefront
    constexpr const auto x_range = std::views::iota(0, max_wavefront);
    const int            nbx     = nx / blocksize_x;
    const int            nby     = ny / blocksize_y;

#pragma unroll
    for (int bwavefront = 0; bwavefront < nby + nbx - 1; bwavefront++) {
        const auto [bxmin, bxmax] = wavefront_coordinates(nby, nbx, bwavefront, 0);
        const auto bx_range       = std::views::iota(bxmin, bxmax + 1);
        auto       idxs           = std::views::cartesian_product(bx_range, x_range);
#pragma unroll
        for (int wavefront = 0; wavefront < blocksize_x + blocksize_y - 1; wavefront++) {
            std::for_each(std::execution::par_unseq, idxs.begin(), idxs.end(), [=](auto pair) {
                const auto [bx, x] = pair;
                const auto by      = bwavefront - bx;
                const uint boundary =
                    (by == (nby - 1)) << 3 | (bx == (nbx - 1)) << 2 | (by == 0) << 1 | (bx == 0);
                const auto [xmin, xmax] =
                    wavefront_coordinates(blocksize_y, blocksize_x, wavefront, boundary);

                if (x >= xmin && x <= xmax) {
                    const int y      = wavefront - x;
                    const int startx = bx * blocksize_x;
                    const int starty = by * blocksize_y;
                    const int x_     = startx + x;
                    const int y_     = starty + y;
                    pnew(y_, x_)     = 0.25 * (pnew(y_ - 1, x_) + pnew(y_, x_ - 1) + p(y_ + 1, x_) +
                                           p(y_, x_ + 1));
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

template <typename T, typename Function>
inline void runAndTimeGaussSeidel(grid<T>& p, grid<T>& pnew, int n, Function gaussSeidelFunc,
                                  std::string name) {
    initialization<T>(p);
    initialization<T>(pnew);

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n; ++it) {
        gaussSeidelFunc(p, pnew);  // Call the specific Gauss-Seidel function
        std::swap(p, pnew);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto grid_size = static_cast<double>(p.size() * sizeof(T) * 2) * 1e-9;  // GB
    auto memory_bw = grid_size * static_cast<double>(n) /
                     std::chrono::duration<double>(end - start).count();  // GB/s

    std::cout << name << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms  " << memory_bw << " GB/s\n";
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations>" << std::endl;
        return 1;
    }

    using type = double;

    const int           ny          = std::stoi(argv[1]);
    const int           nx          = std::stoi(argv[2]);
    const int           n           = std::stoi(argv[3]);
    constexpr const int blocksize_y = 16;
    constexpr const int blocksize_x = 16;

    assert((ny % blocksize_y == 0 && "ny must be divisible by blocksize"));
    assert((nx % blocksize_x == 0 && "nx must be divisible by blocksize"));

    std::vector<type> p_data(ny * nx), pnew_data(ny * nx);

    grid<type> p(p_data.data(), ny, nx);
    grid<type> pnew(pnew_data.data(), ny, nx);

#ifdef SERIAL
    runAndTimeGaussSeidel(p, pnew, n, gauss_seidel<type>, "Serial");
#elif defined WAVE
    runAndTimeGaussSeidel(p, pnew, n, gauss_seidel_wave<type>, "Wave");
#elif defined WAVE2
    runAndTimeGaussSeidel(p, pnew, n, gauss_seidel_wave2<type>, "Wave2");
#elif defined BLOCK_WAVE
    runAndTimeGaussSeidel(p, pnew, n, gauss_seidel_block_wave<type, blocksize_y, blocksize_x>,
                          "Block Wave");
#elif defined BLOCK_WAVE2
    runAndTimeGaussSeidel(p, pnew, n, gauss_seidel_block_wave_2<type, blocksize_y, blocksize_x>,
                          "Block Wave 2")
#else
#error MISSING MACRO DEFINTION TO CHOOSE WAVEFRONT TYPE(SERIAL, WAVE, WAVE2, BLOCK_WAVE, BLOCK_WAVE2)
#endif

    return 0;
}

// # count number of floating point operations and compare
// # look in the kernel and see much memory is being read.
// # talk about perfect cache model vs real cache model.
// # start writing a paragraph every day.
