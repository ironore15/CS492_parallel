#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NANO 1000000000
#define PGSIZE 0x1000
#define SUBSIZE 1000
// #define BLOCK 128
#define GRID 32

int size;
float *matrixA, *vectorB, *vectorX;

__global__
void cudaPivot(float *A_d, int *max_pivot, int pivot, int size)
{
    float max = A_d[pivot * size + pivot];
    int maxi = pivot;
    for (int i = pivot + 1; pivot < size; pivot++)
    {
        float abs = fabsf(A_d[i * size + pivot]);
        if (abs > max)
        {
            max = abs;
            maxi = i;
        }
    }
    *max_pivot = maxi;
    return;
}

__global__
void cudaSwap(float *A_d, int *max_pivot, int pivot, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pivot <= i && i < size)
    {
        float temp = A_d[pivot * size + i];
        A_d[pivot * size + i] = A_d[*max_pivot * size + i];
        A_d[*max_pivot * size + i] = temp;
    }
    return;
}

__global__
void cudaGauss(float *A_d, float *B_d, float *C_d, int x, int y, int n, int size)
{
    int i = n * x + blockIdx.x * blockDim.x + threadIdx.x;
    int j = n * y + blockIdx.y * blockDim.y + threadIdx.y;
    float value = 0.0;

    if (i >= n * (x + 1) || j >= n * (y + 1))
        return;

    for (int k = 0; k < size; k++)
        value += A_d[i * size + k] * B_d[k * size + j];

    C_d[i * size + j] = value;
    return;
}

float *make_vector(int size)
{
    float *vector = (float *) malloc(sizeof(float) * size);
    if (vector == NULL)
    {
        perror("malloc");
        exit(0);
    }

    if (malloc(PGSIZE) == NULL)
    {
        perror("malloc");
        exit(0);
    }
    return vector;
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

void set_vector(float *vector, int size)
{
    for (int i = 0; i < size; i++)
        vector[i] = (float) drand48();
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

void cuda_gaussian(float *A, float *B, float *X, int size)
{
    int mat_size = sizeof(float) * size * size;
    int vec_size = sizeof(float) * size;
    int block_size = (size / GRID) + (size % GRID == 0) ? 0 : 1;
    int *max_pivot;
    float *A_d, *B_d;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(GRID, GRID);

    cudaMalloc((void **) &A_d, mat_size);
    cudaMemcpy(A_d, A, mat_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &B_d, vec_size);
    cudaMemcpy(B_d, B, vec_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &max_pivot, sizeof(int));

    for (int pivot = 0; pivot < size; pivot++)
    {
        cudaPivot<<<1, 1>>>(A_d, max_pivot, pivot, size);
        cudaSwap<<<1, size>>>(A_d, max_pivot, pivot, size);
    }

    cudaMemcpy(X, B_d, vec_size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
}

float L2_norm(float *matrixA, float *vectorX, float *vectorB)
{
    float norm = 0.0, sum;
    for (int i = 0; i < size; i++)
    {
        sum = 0.0;
        for (int j = 0; j < size; j++)
            sum += matrixA[i * size + j] * vectorX[j];
        sum -= vectorB[i];
        norm += sum * sum;
    }
    return sqrt(norm);
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
    vectorB = make_vector(size);
    vectorX = make_vector(size);

    srand48(time(NULL));
    set_matrix(matrixA, size);
    set_vector(vectorB, size);

    printf("Multi Thread Computation Start\n");

    if (clock_gettime(CLOCK_MONOTONIC, &tstart) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    cuda_gaussian(matrixA, vectorB, vectorX, size);

    if (clock_gettime(CLOCK_MONOTONIC, &tend) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    long start_nsec = tstart.tv_sec * NANO + tstart.tv_nsec;
    long end_nsec = tend.tv_sec * NANO + tend.tv_nsec;
    double microsec = (end_nsec - start_nsec) / 1000.0;

    printf("Multi Thread Computation End: %.3f us.\n", microsec);


    printf("L2-norm : %g\n", L2_norm(matrixA, vectorX, vectorB));

    return 0;
}
