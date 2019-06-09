#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NANO 1000000000
#define PGSIZE 0x1000
#define BLOCK 32

int size;
float *matrixA, *matrixB, *matrixBT, *matrixC_serial, *matrixC_cuda;

__global__
void cudaMatMul(float *A_d, float *B_d, float *C_d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float value = 0.0;

    if (i >= n || j >= n)
        return;

    for (int k = 0; k < n; k++)
        value += A_d[i * n + k] * B_d[k * n + j];

    C_d[i * n + j] = value;
    return;
}

float *make_matrix(int size)
{
    float *matrix = (float *) malloc(sizeof(float) * size * size);
    if (matrix == NULL)
    {
        perror("malloc");
        exit(0);
    }

    if (malloc(PGSIZE) == NULL)
    {
        perror("malloc");
        exit(0);
    }
    return matrix;
}

void set_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            matrix[i * size + j] = (float) drand48();
    }
}

void print_matrix(double *matrix, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            printf(" %f", matrix[i * size + j]);
        printf(";");
    }
    printf(" ]\n");
}

void cuda_mmul(float *A, float *B, float *C, int size)
{
    int mem_size = sizeof(float) * size * size;
    int grid_size = (size - 1) / BLOCK + 1;
    float *A_d, *B_d, *C_d;
    dim3 dimBlock(BLOCK, BLOCK);
    dim3 dimGrid(grid_size, grid_size);

    cudaMalloc((void **) &A_d, mem_size);
    cudaMemcpy(A_d, A, mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &B_d, mem_size);
    cudaMemcpy(B_d, B, mem_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &C_d, mem_size);

    cudaMatMul<<<dimBlock, dimGrid>>>(A_d, B_d, C_d, size);

    cudaMemcpy(C, C_d, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void serial_mmul(float *matrixA, float *matrixB, float *matrixC)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixC[i * size + j] = 0.0;
            for (int k = 0; k < size; k++)
                matrixC[i * size + j] += matrixA[i * size + k] * matrixB[k * size + j];
        }
    }
}

int main(int argc, char **argv, char **envp)
{
    int opt;
    struct timespec tstart, tend;

    while ((opt = getopt(argc, argv, "n:p:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                size = atoi(optarg);
                break;
            case '?':
                printf("Usage: %s -n N\n", argv[0]);
                exit(0);
        }
    }

    if (size <= 0)
    {
        printf("Usage: %s -n N\n", argv[0]);
        exit(0);
    }

    matrixA = make_matrix(size);
    matrixB = make_matrix(size);
    matrixC_serial = make_matrix(size);
    matrixC_cuda = make_matrix(size);

    srand48(time(NULL));
    set_matrix(matrixA, size);
    set_matrix(matrixB, size);

    printf("Multi Thread Computation Start\n");

    if (clock_gettime(CLOCK_MONOTONIC, &tstart) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    matrixBT = make_matrix(size);

    cuda_mmul(matrixA, matrixB, matrixC_cuda, size);

    if (clock_gettime(CLOCK_MONOTONIC, &tend) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    long start_nsec = tstart.tv_sec * NANO + tstart.tv_nsec;
    long end_nsec = tend.tv_sec * NANO + tend.tv_nsec;
    double microsec = (end_nsec - start_nsec) / 1000.0;

    printf("Multi Thread Computation End: %.3f us.\n", microsec);

    printf("Single Thread Computation Start\n");

    if (clock_gettime(CLOCK_MONOTONIC, &tstart) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    serial_mmul(matrixA, matrixB, matrixC_serial);

    if (clock_gettime(CLOCK_MONOTONIC, &tend) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    start_nsec = tstart.tv_sec * NANO + tstart.tv_nsec;
    end_nsec = tend.tv_sec * NANO + tend.tv_nsec;
    microsec = (end_nsec - start_nsec) / 1000.0;

    printf("Single Thread Computation End: %.3f us.\n", microsec);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (fabs(matrixC_cuda[i * size + j] - matrixC_serial[i * size + j]) > 1e-3)
            {
                printf("Verification Fail.\n");
                printf("(%d, %d): %f - %f\n", i, j, matrixC_cuda[i * size + j], matrixC_serial[i * size + j]);
                exit(0);
            }
        }
    }
    printf("Verification Success.\n");


    // print_matrix(matrixA, size);
    // print_matrix(matrixB, size);
    // print_matrix(matrixC, size);

    return 0;
}
