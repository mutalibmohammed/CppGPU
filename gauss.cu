#include <iostream>



template <typename T>
__global__
void gauss_seidel(int ny, int nx, const T *p, T *pnew) {
    for (int i = 1; i < ny - 1; i++) {
        for (int j = 1; j < nx - 1; j++) {
            printf("Do I xome here\n");
            pnew[i * nx + j] = 0.25 * (pnew[(i - 1) * nx + j] + pnew[i * nx + (j - 1)] + p[(i + 1) * nx + j] + p[i * nx + (j + 1)]);
        }
    }
}

int main() {

    typedef double gtype;

    constexpr const int iterations = 1000;
    constexpr const int nx = 101;
    constexpr const int ny = 101;
 
    auto p = new gtype[ny][nx]();
    auto pnew = new gtype[ny][nx]();
 
    gtype *d_p;
    gtype *d_pnew;

    cudaError_t err;

    err = cudaMalloc((void **) &d_p, ny * nx * sizeof(gtype));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &d_pnew, ny * nx * sizeof(gtype));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;


    for (int i = 1; i < ny - 1; i++)
    {
        for (int j = 1; j < nx - 1; j++)
        {
            p[i][j] = 1;
            pnew[i][j] = 1;
        }
    }

    
    err = cudaMemcpy(d_p, p, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(d_pnew, pnew, ny * nx * sizeof(gtype), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;


    for (int i = 0; i < iterations; i++)
    {
        gauss_seidel<<<1, 1>>>(ny, nx, (i % 2 == 0) ? d_p: d_pnew, (i % 2 == 0) ? d_pnew : d_p);
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    }

    err = cudaMemcpy(p, d_p, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(pnew, d_pnew, ny * nx * sizeof(gtype), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;


    delete[] p;
    delete[] pnew;

    cudaFree(d_p);
    cudaFree(d_pnew);
    return 0;
}