#include <stdexec/execution.hpp>
#include <exec/on.hpp>
#include <exec/static_thread_pool.hpp>
#include <nvexec/stream_context.cuh>
#include <vector>

int main()
{
    // Declare a pool of 8 worker CPU threads:
    exec::static_thread_pool pool(8);

    // Declare a GPU stream context:
    nvexec::stream_context stream_ctx{};

    // Get a handle to the thread pool:
    auto cpu_sched = pool.get_scheduler();
    auto gpu_sched = stream_ctx.get_scheduler();

    // Declare three dynamic array with N elements
    std::size_t N = 5;
    std::vector<int> v0{1, 1, 1, 1, 1};
    std::vector<int> v1{2, 2, 2, 2, 2};
    std::vector<int> v2{0, 0, 0, 0, 0};

    // Describe some work:
    auto work = stdexec::when_all(
                    // Double v0 on the CPU
                    stdexec::just() | exec::on(cpu_sched, stdexec::bulk(N, [v0 = v0.data()](std::size_t i)
                                                                        { v0[i] *= 2; })),
                    // Triple v1 on the GPU
                    stdexec::just() | exec::on(gpu_sched, stdexec::bulk(N, [v1 = v1.data()](std::size_t i)
                                                                        { v1[i] *= 3; }))) |
                stdexec::transfer(cpu_sched)
                // Add the two vectors into the output vector v2 = v0 + v1:
                | stdexec::bulk(N, [&](std::size_t i)
                                { v2[i] = v0[i] + v1[i]; }) |
                stdexec::then([&]
                              { 
        int r = 0;
        for (std::size_t i = 0; i < N; ++i) r += v2[i];
        return r; });

        auto [sum] = stdexec::sync_wait(work).value();

    // Print the results:
    std::printf("sum = %d\n", sum);
    for (int i = 0; i < N; ++i)
    {
        std::printf("v0[%d] = %d, v1[%d] = %d, v2[%d] = %d\n", i, v0[i], i, v1[i], i, v2[i]);
    }
    return 0;
}