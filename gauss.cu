#include <iostream>

__global__
void gauss_seidel(int ncellx, int ncelly, int nu, double *v1, double *v2, double *an, double *as, double *aw, double *ae)
{
    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    // int j = blockIdx.y*blockDim.y + threadIdx.y;
    // int k = blockIdx.z*blockDim.z + threadIdx.z;

    for(int i = 1; i < ncelly - 1; i++) {
        for(int j = 1; j < ncellx - 1; j++) {
            for(int k = 0; k < nu; k++) {
                v2[i * ncellx * nu + j * nu + k] = 0;
                for(int l = 0; l < nu; l++) {
                    v2[i * ncellx * nu + j * nu + k] += 
                        aw[i * ncellx * nu * nu + j * nu * nu + k * nu + l] * v2[(i - 1)  * ncellx * nu + j * nu + l] +
                        an[i * ncellx * nu * nu + j * nu * nu + k * nu + l] * v2[i * ncellx * nu + (j - 1) * nu + l] +

                        ae[i * ncellx * nu * nu + j * nu * nu + k * nu + l] * v1[(i + 1) * ncellx * nu + j * nu + l] +
                        as[i * ncellx * nu * nu + j * nu * nu + k * nu + l] * v1[i * ncellx * nu + (j + 1) * nu + l];
                }
            }
        }
     }
}


int main() {
    constexpr const int n = 10;
    constexpr const int ncellx = 1000;
    constexpr const int ncelly = 1000;
    constexpr const int nu = 10;


    auto v1 = new double[ncelly][ncellx][nu];
    auto v2 = new double[ncelly][ncellx][nu];
    auto an = new double[ncelly][ncellx][nu][nu]();
    auto as = new double[ncelly][ncellx][nu][nu]();
    auto aw = new double[ncelly][ncellx][nu][nu]();
    auto ae = new double[ncelly][ncellx][nu][nu]();

    double *dv1;
    double *dv2;
    double *dan;
    double *das;
    double *daw;
    double *dae;
    

    cudaError_t err;

    err = cudaMalloc((void **) &dv1, ncelly * ncellx * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &dv2, ncelly * ncellx * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &dan, ncelly * ncellx * nu * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &das, ncelly * ncellx * nu * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &daw, ncelly * ncellx * nu * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMalloc((void **) &dae, ncelly * ncellx * nu * nu * sizeof(double));
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;


    for (int i = 1; i < ncelly - 1; i++)
    {
        for (int j = 1; j < ncellx - 1; j++)
        {
            for (int k = 0; k < nu; k++)
            {
                v1[i][j][k] = 1.;
                v2[i][j][k] = 1.;
                an[i][j][k][k] = 0.25;
                as[i][j][k][k] = 0.25;
                aw[i][j][k][k] = 0.25;
                ae[i][j][k][k] = 0.25;
            }
        }
    }

    
    for (int i = 0; i < ncelly; i++)
    {
        for (int k = 0; k < nu; k++)
        {
            v1[i][0][k] = 0.;
            v1[i][ncellx - 1][k] = 0.;
            v2[i][0][k] = 0.;
            v2[i][ncellx - 1][k] = 0.;
        }
    }

    err = cudaMemcpy(dv1, v1, ncelly * ncellx * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(dv2, v2, ncelly * ncellx * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(dan, an, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(das, as, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(daw, aw, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(dae, ae, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;


    for (int i = 0; i < n; i++)
    {
        gauss_seidel<<<1, 1>>>(ncellx, ncelly, nu, (i % 2 == 0) ? dv1 : dv2, (i % 2 == 0) ? dv2 : dv1, dan, das, daw, dae);
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    }

    err = cudaMemcpy(v1, dv1, ncelly * ncellx * nu * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(v2, dv2, ncelly * ncellx * nu * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    // err = cudaMemcpy(an, dan, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    // err = cudaMemcpy(as, das, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    // err = cudaMemcpy(aw, daw, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    // err = cudaMemcpy(ae, dae, ncelly * ncellx * nu * nu * sizeof(double), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;

    double sum = 0;
    for (int i = 1; i < ncelly - 1; i++)
    {
        for (int j = 1; j < ncellx -1; j++)
        {
            for (int k = 0; k < nu; k++)
            {
                sum += v1[i][j][k] * v1[i][j][k];
            }
        }
    }

    std::cout << sum << std::endl;


    double sum2 = 0;
    for (int i = 1; i < ncelly - 1; i++)
    {
        for (int j = 1; j < ncellx -1; j++)
        {
            for (int k = 0; k < nu; k++)
            {
                sum2 += v2[i][j][k] * v2[i][j][k];
            }
        }
    }

    std::cout << sum2 << std::endl;

    delete[] v1;
    delete[] v2;
    delete[] an;
    delete[] as;
    delete[] aw;
    delete[] ae;

    cudaFree(dv1);
    cudaFree(dv2);
    cudaFree(dan);
    cudaFree(das);
    cudaFree(daw);
    cudaFree(dae);

    return 0;
}