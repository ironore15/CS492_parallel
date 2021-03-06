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
    int start;
    int end;
};

int size;
double **matrixA, **matrixB, **matrixBT, **matrixC_serial, **matrixC_multiple;
pthread_barrier_t transpose_barrier;

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

void trans_routine(double **matrix, double **matrixT, int start, int end)
{
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < size; j++)
            matrixT[i][j] = matrix[j][i];
    }
}

void mmul_routine(double **matrixA, double **matrixB, double **matrixC, int start, int end)
{
    double sum;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < size; j++)
        {
            sum = 0.0;
            for (int k = 0; k < size; k++)
                sum += matrixA[i][k] * matrixB[j][k];
            matrixC[i][j] = sum;
        }
    }
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

void *thread_routine(void *bound)
{
    struct param *param = (struct param *) bound;
    int start = param->start;
    int end = param->end;

    trans_routine(matrixB, matrixBT, start, end);
    pthread_barrier_wait(&transpose_barrier);

    mmul_routine(matrixA, matrixBT, matrixC_multiple, start, end);
    return NULL;
}

void serial_mmul(double **matrixA, double **matrixB, double **matrixC)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixC[i][j] = 0.0;
            for (int k = 0; k < size; k++)
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
        }
    }
}

int main(int argc, char **argv, char **envp)
{
    int opt, numThreads = 0;
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
    matrixB = make_matrix(size);
    matrixC_serial = make_matrix(size);
    matrixC_multiple = make_matrix(size);

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

    pthread_t *threads = (pthread_t *) malloc((numThreads - 1) * sizeof(pthread_t));
    struct param *bounds = (struct param *) malloc(sizeof(struct param) * numThreads);

    pthread_barrier_init(&transpose_barrier, NULL, numThreads);

    for (int i = 0; i < numThreads; i++)
    {
        if (i == 0)
            bounds[i].start = 0;
        else
            bounds[i].start = bounds[i-1].end;

        if (i == numThreads - 1)
            bounds[i].end = size;
        else
            bounds[i].end = bounds[i].start + amount;
    }

    for (int i = 0; i < numThreads - 1; i++)
        pthread_create(&threads[i], NULL, thread_routine, (void *) &bounds[i]);
    thread_routine(&bounds[numThreads - 1]);

    for (int i = 0; i < numThreads - 1; i++)
        pthread_join(threads[i], NULL);

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
            if (fabs(matrixC_serial[i][j] - matrixC_multiple[i][j]) > 1e-10)
            {
                printf("Verification Fail.\n");
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
