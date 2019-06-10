#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NANO 1000000000
#define PGSIZE 0x1000
#define BLOCK1 1024
#define BLOCK2 32

int size;
float *matrixA, *vectorB, *vectorX;

__global__
void cudaPivot(float *A_d, int *max_pivot, int pivot, int size)
{
    float max = fabsf(A_d[pivot * size + pivot]);
    int maxi = pivot;
    for (int i = pivot + 1; i < size; i++)
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
void cudaSwap(float *A_d, float *B_d, int *max_pivot, int pivot, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((pivot <= i) && (i < size))
    {
        float temp = A_d[pivot * size + i];
        A_d[pivot * size + i] = A_d[*max_pivot * size + i];
        A_d[*max_pivot * size + i] = temp;
    }
    else if (i == size)
    {
        float temp = B_d[pivot];
        B_d[pivot] = B_d[*max_pivot];
        B_d[*max_pivot] = temp;
    }
    return;
}

__global__
void cudaCalcM(float *A_d, float *B_d, float *M_d, int pivot, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((pivot < i) && (i < size))
    {
        M_d[i] = A_d[i * size + pivot] / A_d[pivot * size + pivot];
    }
}

__global__
void cudaGauss(float *A_d, float *B_d, float *M_d, int pivot, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((pivot < i) && (i < size) && (pivot <= j) && (j <= size))
    {
        if (j == size)
        {
            B_d[i] -= M_d[i] * B_d[pivot];
        }
        else
        {
            A_d[i * size + j] -= M_d[i] * A_d[pivot * size + j];
        }
    }
}

__global__
void cudaBackSub(float *A_d, float *B_d, int pivot, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < pivot)
    {
	float m = B_d[pivot] / A_d[pivot * size + pivot];
        B_d[i] -= m * A_d[i * size + pivot];
    }
}

__global__
void cudaCoeff(float *A_d, float *B_d, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        B_d[i] = B_d[i] / A_d[i * size + i];
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

void print_vector(float *vector, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        printf(" %f", vector[i]);
        printf(";");
    }
    printf(" ]\n");
}

void print_matrix(float *matrix, int size)
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
    int grid1 = ((size + 1) / BLOCK1) + (((size + 1) % BLOCK1 == 0) ? 0 : 1);
    int grid2 = ((size + 1) / BLOCK2) + (((size + 1) % BLOCK2 == 0) ? 0 : 1);
    int *max_pivot;
    float *A_d, *B_d, *M_d;
    dim3 dimBlock(BLOCK2, BLOCK2);
    dim3 dimGrid(grid2, grid2);

    cudaMalloc((void **) &A_d, mat_size);
    cudaMemcpy(A_d, A, mat_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &B_d, vec_size);
    cudaMemcpy(B_d, B, vec_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &M_d, vec_size);
    cudaMalloc((void **) &max_pivot, sizeof(int));

    for (int pivot = 0; pivot < size; pivot++)
    {
        cudaPivot<<<1, 1>>>(A_d, max_pivot, pivot, size);

        cudaSwap<<<grid1, BLOCK1>>>(A_d, B_d, max_pivot, pivot, size);

	cudaCalcM<<<grid1, BLOCK1>>>(A_d, B_d, M_d, pivot, size);

	cudaGauss<<<dimGrid, dimBlock>>>(A_d, B_d, M_d, pivot, size);
    }

    for (int pivot = size - 1; pivot >= 0; pivot--)
    {
    	cudaBackSub<<<grid1, BLOCK1>>>(A_d, B_d, pivot, size);
    }

    cudaCoeff<<<grid1, BLOCK1>>>(A_d, B_d, size);

    cudaMemcpy(X, B_d, vec_size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(M_d);
    cudaFree(max_pivot);
}

int pivot_routine(float *matrix, int start, int end, int pivot)
{
    float t_max = 0.0, max = 0.0;
    int t_maxi = pivot, maxi = pivot;
    {
        for (int i = start; i < end; i++)
        {
            float abs = fabsf(matrix[i * size + pivot]);
            if (abs > max)
            {
                max = abs;
                maxi = i;
            }
        }

        if (max > t_max)
        {
            t_max = max;
            t_maxi = maxi;
        }
    }
    return t_maxi;
}

void gauss_routine(float *matrix, float *vector, int start, int end, int pivot)
{
    for (int i = start; i < end; i++)
    {
        float m = matrix[i * size + pivot] / matrix[pivot * size + pivot];

        for (int j = pivot; j < size; j++)
            matrix[i * size + j] -= m * matrix[pivot * size + j];
        vector[i] -= m * vector[pivot];
    }
}

void back_routine(float *matrix, float *vector, int start, int end, int pivot)
{
    float m = vector[pivot] / matrix[pivot * size + pivot];
    for (int i = start; i < end; i++)
        vector[i] -= m * matrix[i * size + pivot];
}

void single_gaussian(void)
{
    int maxi;
    float swap;

    for (int i = 0; i < size; i++)
    {
        maxi = pivot_routine(matrixA, i, size, i);

	for (int j = 0; j < size; j++)
        {
            swap = matrixA[i * size + j];
            matrixA[i * size + j] = matrixA[maxi * size + j];
            matrixA[maxi * size + j] = swap;
        }

        swap = vectorB[i];
        vectorB[i] = vectorB[maxi];
        vectorB[maxi] = swap;

        gauss_routine(matrixA, vectorB, i + 1, size, i);
    }

    for (int i = size - 1; i >= 0; i--)
    {
        back_routine(matrixA, vectorB, 0, i, i);
        vectorB[i] /= matrixA[i * size + i];
    }

    return;
}


float L2_norm(float *matrixA, float *vectorX, float *vectorB, int size)
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
    int opt, t_sec = time(NULL);
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

    srand48(t_sec);
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

    printf("L2-norm : %g\n", L2_norm(matrixA, vectorX, vectorB, size));


    printf("Single Thread Computation Start\n");

    if (clock_gettime(CLOCK_MONOTONIC, &tstart) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    single_gaussian();

    if (clock_gettime(CLOCK_MONOTONIC, &tend) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    start_nsec = tstart.tv_sec * NANO + tstart.tv_nsec;
    end_nsec = tend.tv_sec * NANO + tend.tv_nsec;
    microsec = (end_nsec - start_nsec) / 1000.0;

    printf("Single Thread Computation End: %.3f us.\n", microsec);

    free(vectorX);
    vectorX = vectorB;

    free(matrixA);
    free(vectorB);
    matrixA = make_matrix(size);
    vectorB = make_vector(size);

    srand48(t_sec);
    set_matrix(matrixA, size);
    set_vector(vectorB, size);

    printf("L2-norm : %g\n", L2_norm(matrixA, vectorX, vectorB, size));

    return 0;
}
