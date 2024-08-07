#include "stdexec/execution.hpp"
#include "exec/on.hpp"
#include "exec/static_thread_pool.hpp"
#include "nvexec/stream_context.cuh"
#include "exec/repeat_n.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

template <typename T>
constexpr std::pair<T, T> wavefront_coordinates(T ny, T nx, T wavefront, uint boundary)
{
    int left = boundary & 1;
    int top = (boundary >> 1) & 1;
    int right = (boundary >> 2) & 1;
    int bottom = (boundary >> 3) & 1;

    T xmin = std::max(left, wavefront - ((ny - 1 - bottom)));
    T xmax = std::min(wavefront - top, nx - 1 - right);
    return {xmin, xmax >= xmin ? xmax : xmin - 1};
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Error incorrect arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> <iterations>" << std::endl;
        return 1;
    }

    using type = double;

    const int ny = std::stoi(argv[1]);
    const int nx = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);

    std::vector<type> p_data(ny * nx), pnew_data(ny * nx);

    // grid<type> p(p_data.data(), ny, nx);
    // grid<type> pnew(pnew_data.data(), ny, nx);

    // initialization(p);
    // initialization(pnew);

    // Declare a GPU stream context:
    nvexec::stream_context stream_ctx{};

    // Get the GPU scheduler:
    auto gpu_sched = stream_ctx.get_scheduler();

    stdexec::sync_wait(stdexec::when_all(
        stdexec::just() | exec::on(gpu_sched, stdexec::bulk(ny * nx, [p_data = p_data.data(), ny, nx](std::size_t i)
                                                            { p_data[i] = 1.0; })),
        stdexec::just() | exec::on(gpu_sched, stdexec::bulk(ny * nx, [pnew_data = pnew_data.data(), ny, nx](std::size_t i)
                                                            { pnew_data[i] = 1.0; }))));

    // TODO reduction
    // Describe some work:

    int nwavefronts = ny + nx - 1;
    for (size_t it = 0; it < n; it++)
    {
        int *wavefront = new int;
        *wavefront = 0;

        auto work = stdexec::just() | exec::on(gpu_sched, stdexec::bulk(ny * nx, [pnew_data = pnew_data.data(), p_data = p_data.data(), ny, nx, wavefront](std::size_t i)
                                                                        {
        auto [xmin, xmax] = wavefront_coordinates(ny, nx, *wavefront, 0b1111);
        int ymin = *wavefront - xmin;
        int ymax = *wavefront - xmax;

        if (i >= ymin * nx + xmin && i <= ymax * nx + xmax)
        {
            pnew_data[i] = 0.25 * (pnew_data[i - nx] + pnew_data[i - 1] + p_data[i + nx] + p_data[i + 1]);
        } 
        *wavefront += 1; })) |
                    exec::repeat_n(nwavefronts);
        stdexec::sync_wait(std::move(work));

        printf("wavefront: %d\n", *wavefront);
        delete wavefront;
    }

    type sum = 0.f;
    for (int i = 0; i < ny * nx; i++)
    {
        sum += pnew_data[i];
    }

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}