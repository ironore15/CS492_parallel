#include "mmreader.hpp"
#include <time.h>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#define THREAD 6

struct param
{
    struct sparse_mtx *A;
    struct dense_mtx *B;
    struct dense_mtx *C;
    int start;
    int end;
};

bool
SCsrMatrixfromFile(struct sparse_mtx *A, const char* filePath)
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
        {
            std::cout << "Reading file name error" << std::endl;
            return false;
        }
    }
    else
        return false;

    // Read data from a file on disk into buffers
    // Data is read natively as COO format with the reader
    MatrixMarketReader mm_reader;
    if( mm_reader.MMReadFormat(filePath) )
        return false;

    // JPA: Shouldn't that just be an assertion check? It seems to me that
    // the user have to call clsparseHeaderfromFile before calling this function,
    // otherwise the whole pCsrMatrix will be broken;
    A->nrow = mm_reader.GetNumRows( );
    A->ncol = mm_reader.GetNumCols( );
    A->nnze = mm_reader.GetNumNonZeroes( );

    A->row = (int32_t *)malloc((A->nrow + 1) * sizeof(int32_t));
    A->val = (float *)malloc(A->nnze * sizeof(float));
    A->col = (int32_t *)malloc(A->nnze * sizeof(int32_t));

    if(A->row == NULL || A->col == NULL || A->val == NULL)
    {
        if(A->row == NULL)
            free((void *)A->row);
        if(A->col == NULL)
            free((void *)A->col);
        if(A->val == NULL)
            free((void *)A->val);
        return false;
    }

    //  The following section of code converts the sparse format from COO to CSR
    Coordinate* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + A->nnze, CoordinateCompare );

    int32_t current_row = 1;

    A->row[ 0 ] = 0;

    for (int32_t i = 0; i < A->nnze; i++)
    {
        A->col[ i ] = coords[ i ].y;
        A->val[ i ] = coords[ i ].val;

        while( coords[ i ].x >= current_row )
            A->row[ current_row++ ] = i;
    }

    A->row[ current_row ] = A->nnze;

    while( current_row <= A->nrow )
        A->row[ current_row++ ] = A->nnze;

    return true;
}

void multiply_single(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C)
{
    C->nrow = A->nrow;
    C->ncol = B->ncol;
    C->val = (float *)calloc(C->nrow * C->ncol, sizeof(float));

    for (int row = 0; row < A->nrow; row++)
    {
        int32_t start = A->row[row];
        int32_t end = A->row[row + 1];

        if (start == A->nnze)
            break;

        for (int i = start; i < end; i++)
        {
            int32_t col = A->col[i];
            float val = A->val[i];

            for (int j = 0; j < B->ncol; j++)
                C->val[C->ncol * row + j] += val * B->val[B->ncol * col + j];
        }
    }
}

void multiply_openmp(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C)
{
    C->nrow = A->nrow;
    C->ncol = B->ncol;
    C->val = (float *)calloc(C->nrow * C->ncol, sizeof(float));

    #pragma omp parallel for schedule(guided, 16) num_threads(THREAD)
    for (int row = 0; row < A->nrow; row++)
    {
        int32_t start = A->row[row];
        int32_t end = A->row[row + 1];

        for (int i = start; i < end; i++)
        {
            int32_t col = A->col[i];
            float val = A->val[i];

            for (int j = 0; j < B->ncol; j++)
                C->val[C->ncol * row + j] += val * B->val[B->ncol * col + j];
        }
    }
}

int compare_matrix(struct dense_mtx *C1, struct dense_mtx *C2)
{
    if (C1->nrow != C2->nrow || C1->ncol != C2->ncol)
        return 1;

    for (int i = 0; i < C1->nrow; i++)
    {
        for (int j = 0; j < C1->ncol; j++)
        {
            if (C1->val[C1->ncol * i + j] != C2->val[C2->ncol * i + j])
                return 1;
        }
    }
    return 0;
}

uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

int main(int argc, char **argv)
{
    struct sparse_mtx A;
    if(!SCsrMatrixfromFile(&A, argv[1]))
    {
        std::cout << "read failed." << std::endl;
        return 0;
    }

    struct dense_mtx B;
    B.nrow = A.ncol;
    B.ncol = atoi(argv[2]);
    if(B.ncol < 0)
    {
        free(A.row);
        free(A.col);
        free(A.val);
        std::cerr << "Invalid argument for the number of columns of B." << std::endl;
    }
    B.val = (float *)malloc(sizeof(float) * B.nrow * B.ncol);

    srand((unsigned int)time(NULL));
    for(int i = 0; i < B.nrow; i++)
    {
        for(int j = 0; j < B.ncol; j++)
        {
            B.val[B.ncol * i + j] = ((float)rand()/(float)(RAND_MAX)) * ((rand() % 2) ? 1.0f : -1.0f);
        }
    }

    struct dense_mtx C1, C2;
    C1.val = NULL;
    C2.val = NULL;

    uint64_t start, end;

    std::cout << "Single Thread Computation Start" << std::endl;
    start = GetTimeStamp();
    multiply_single(&A, &B, &C1);
    end = GetTimeStamp();
    std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;
    std::cout << "Multi Thread Computation Start" << std::endl;
    start = GetTimeStamp();
    multiply_openmp(&A, &B, &C2);
    end = GetTimeStamp();
    std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

    if (compare_matrix(&C1, &C2))
        std::cout << "Verification Fail." << std::endl;
    else
        std::cout << "Verification Success." << std::endl;

    free(A.row);
    free(A.col);
    free(A.val);
    free(B.val);
    if(C1.val != NULL)
        free(C1.val);
    if(C2.val != NULL)
        free(C2.val);

    return 0;
}
