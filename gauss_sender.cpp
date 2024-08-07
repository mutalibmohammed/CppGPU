#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "exec/on.hpp"
#include "exec/repeat_n.hpp"
#include "exec/static_thread_pool.hpp"
#include "nvexec/stream_context.cuh"
#include "stdexec/execution.hpp"

template <typename T>
constexpr std::pair<T, T> wavefront_coordinates(T ny, T nx, T wavefront, uint boundary)
{
    int left = boundary & 1;
    int top = (boundary >> 1) & 1;
    int right = (boundary >> 2) & 1;
    int bottom = (boundary >> 3) & 1;

    T xmin = std::max(left, wavefront - ((ny - 1 - bottom)));
    T xmax = std::min(wavefront - top, nx - 1 - right);
    return {xmin, xmax};
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations>" << std::endl;
        return 1;
    }

    using type = float;

    const int ny = std::stoi(argv[1]);
    const int nx = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);

    assert((nx % 2 == 0));

    std::vector<type> p_data(ny * nx), pnew_data(ny * nx);

    // Declare a GPU stream context:
    nvexec::stream_context stream_ctx{};

    // Get the GPU scheduler:
    auto gpu_sched = stream_ctx.get_scheduler();

    // Initialize the data:
    stdexec::sync_wait(
        stdexec::just() |
        exec::on(gpu_sched,
                 stdexec::bulk(
                     ny * nx,
                     [p_data = p_data.data(), pnew_data = pnew_data.data(), ny, nx](std::size_t i) {
                         p_data[i]    = 0.0;
                         pnew_data[i] = 0.0;
                     }) |

                     stdexec::bulk(ny,
                                   [p_data = p_data.data(), pnew_data = pnew_data.data(), ny,
                                    nx](std::size_t i) {
                                       p_data[i * nx]             = 10.f;
                                       p_data[i * nx + nx - 1]    = 10.f;
                                       pnew_data[i * nx]          = 10.f;
                                       pnew_data[i * nx + nx - 1] = 10.f;
                                   }) |
                     stdexec::bulk(nx, [p_data = p_data.data(), pnew_data = pnew_data.data(), ny,
                                        nx](std::size_t i) {
                         p_data[i]                    = 10.f;
                         p_data[(ny - 1) * nx + i]    = 10.f;
                         pnew_data[i]                 = 10.f;
                         pnew_data[(ny - 1) * nx + i] = 10.f;
                     })));

    // for (int y = 0; y < ny; y++) {
    //     for (int x = 0; x < nx; x++) {
    //         std::printf("%2.3f  ", p_data[y * nx + x]);
    //     }
    //     std::cout << '\n';
    // }

    int *wavefront = new int;
    *wavefront     = 0;

    int nwavefronts = ny + nx - 1;

#pragma unroll
    for (size_t it = 0; it < n; it++) {
        auto work =
            stdexec::just() |
            exec::on(gpu_sched,
                     stdexec::bulk(ny * nx,
                                   [pnew_data = pnew_data.data(), p_data = p_data.data(), ny, nx,
                                    it, wavefront](std::size_t i) {
                                       auto [xmin, xmax] =
                                           wavefront_coordinates(ny, nx, *wavefront, 0b1111);

                                       int ymin = *wavefront - xmin;
                                       int ymax = *wavefront - xmax;

                                       int x = i & (nx - 1);
                                       int y = i / nx;

                                       //    std::printf("wavefront: %d, xmin: %d, xmax: %d, ymin:
                                       //    %d, "
                                       //                "ymax: %d,  %d  <= %lu <= %d , i: %lu\n",
                                       //                *wavefront, xmin, xmax, ymin, ymax, xmin,
                                       //                i & (nx - 1), xmax, i);

                                       if (xmin <= x && x <= xmax && ymin <= y && y <= ymax) {
                                           pnew_data[i] =
                                               0.25 * (pnew_data[i - nx] + pnew_data[i - 1] +
                                                       p_data[i + nx] + p_data[i + 1]);
                                       }
                                   }) |
                         stdexec::then([wavefront]() { *wavefront += 1; })) |
            exec::repeat_n(nwavefronts - 1) | stdexec::then([wavefront]() { *wavefront = 0; });
        stdexec::sync_wait(std::move(work));
        std::swap(p_data, pnew_data);
    }

    // for (int y = 0; y < ny; y++) {
    //     for (int x = 0; x < nx; x++) {
    //         std::printf("%2.3f  ", p_data[y * nx + x]);
    //     }
    //     std::cout << '\n';
    // }

    type sum = 0.f;
    for (int i = 0; i < ny * nx; i++) {
        sum += p_data[i];
    }

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}