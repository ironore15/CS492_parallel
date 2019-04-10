#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NANO 1000000000
#define PGSIZE 0x1000

struct param
{
    int index;
};

int size = 0, numThreads = 0;
double **matrixA, *vectorB, *vectorX;
pthread_barrier_t thread_barrier;

double *make_vector(int size)
{
    double *vector = (double *) malloc(sizeof(double) * size);
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

double **make_matrix(int size)
{
    double **matrix = (double **) malloc(sizeof(double *) * size);
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

    for (int i = 0; i < size; i++)
    {
        matrix[i] = (double *) malloc(sizeof(double) * size);
        if (matrix[i] == NULL)
        {
            perror("malloc");
            exit(0);
        }
    }

    if (malloc(PGSIZE) == NULL)
    {
        perror("malloc");
        exit(0);
    }
    return matrix;
}

void set_matrix(double **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            matrix[i][j] = drand48();
    }
}

void set_vector(double *vector, int size)
{
    for (int i = 0; i < size; i++)
        vector[i] = drand48();
}

int pivot_routine(double **matrix, int start, int end, int pivot)
{
    double abs, max = 0.0;
    int maxi = pivot;
    for (int i = start; i < end; i++)
    {
        abs = fabs(matrix[i][pivot]);
        if (abs > max)
        {
            max = abs;
            maxi = i;
        }
    }
    return maxi;
}

void gauss_routine(double **matrix, double *vector, int start, int end, int pivot)
{
    double m;
    for (int i = start; i < end; i++)
    {
        m = matrix[i][pivot] / matrix[pivot][pivot];

        for (int j = pivot; j < size; j++)
            matrix[i][j] -= m * matrix[pivot][j];
        vector[i] -= m * vector[pivot];
    }
}

void back_routine(double **matrix, double *vector, int start, int end, int pivot)
{
    double m = vector[pivot] / matrix[pivot][pivot];
    for (int i = start; i < end; i++)
        vector[i] -= m * matrix[i][pivot];
}

void print_matrix(double **matrix, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            printf(" %f", matrix[i][j]);
        printf(";");
    }
    printf(" ]\n");
}

void print_vector(double *vector, int size)
{
    printf("[");
    for (int i = 0; i < size; i++)
        printf(" %f", vector[i]);
    printf("; ]\n");
}

double L2_norm(double **matrixA, double *vectorX, double *vectorB)
{
    double norm = 0.0, sum;
    for (int i = 0; i < size; i++)
    {
        sum = 0.0;
        for (int j = 0; j < size; j++)
            sum += matrixA[i][j] * vectorX[j];
        sum -= vectorB[i];
        norm += sum * sum;
    }
    return sqrt(norm);
}

void *thread_routine(void *bound)
{
    struct param *param = (struct param *) bound;
    int index = param->index;
    int start, end, maxi;
    double amount, abs, max, swap, *swap_row;

    for (int i = 0; i < size; i++)
    {
        if (index == 0)
        {
            maxi = pivot_routine(matrixA, i, size, i);

            swap = vectorB[i];
            vectorB[i] = vectorB[maxi];
            vectorB[maxi] = swap;

            swap_row = matrixA[i];
            matrixA[i] = matrixA[maxi];
            matrixA[maxi] = swap_row;
        }
        pthread_barrier_wait(&thread_barrier);

        amount = (double) (size - i - 1) / (double) numThreads;

        start = lround(index * amount) + i + 1;
        end = lround((index + 1) * amount) + i + 1;

        gauss_routine(matrixA, vectorB, start, end, i);
        pthread_barrier_wait(&thread_barrier);
    }

    return NULL;
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
            case 'p':
                numThreads = atoi(optarg);
                break;
            case '?':
                printf("Usage: %s -n N -p N\n", argv[0]);
                exit(0);
        }
    }

    if (size <= 0 || numThreads <= 0)
    {
        printf("Usage: %s -n N -p N\n", argv[0]);
        exit(0);
    }

    int amount = size / numThreads;

    matrixA = make_matrix(size);
    vectorB = make_vector(size);

    srand48(t_sec);
    set_matrix(matrixA, size);
    set_vector(vectorB, size);

    // print_matrix(matrixA, size);
    // print_vector(vectorB, size);

    if (clock_gettime(CLOCK_MONOTONIC, &tstart) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    pthread_t *threads = (pthread_t *) malloc((numThreads - 1) * sizeof(pthread_t));
    struct param *bounds = (struct param *) malloc(sizeof(struct param) * numThreads);

    pthread_barrier_init(&thread_barrier, NULL, numThreads);

    for (int i = 0; i < numThreads; i++)
        bounds[i].index = i;

    for (int i = 0; i < numThreads - 1; i++)
        pthread_create(&threads[i], NULL, thread_routine, (void *) &bounds[i]);
    thread_routine(&bounds[numThreads - 1]);

    for (int i = 0; i < numThreads - 1; i++)
        pthread_join(threads[i], NULL);

    for (int i = size - 1; i >= 0; i--)
    {
        back_routine(matrixA, vectorB, 0, i, i);
        vectorB[i] /= matrixA[i][i];
    }

    if (clock_gettime(CLOCK_MONOTONIC, &tend) == -1)
    {
        perror("clock_gettime");
        exit(0);
    }

    long start_nsec = tstart.tv_sec * NANO + tstart.tv_nsec;
    long end_nsec = tend.tv_sec * NANO + tend.tv_nsec;
    double millisec = (end_nsec - start_nsec) / 1000000.0;

    printf("time: %.5f msec\n", millisec);

    vectorX = vectorB;
    matrixA = make_matrix(size);
    vectorB = make_vector(size);

    srand48(t_sec);
    set_matrix(matrixA, size);
    set_vector(vectorB, size);

    // print_matrix(matrixA, size);
    // print_vector(vectorB, size);
    // print_vector(vectorX, size);

    printf("L2-norm : %g\n", L2_norm(matrixA, vectorX, vectorB));

    return 0;
}
