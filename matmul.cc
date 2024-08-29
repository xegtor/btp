#include<stdio.h>
#include <iostream>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "basic.hpp"

extern void gemm(int m, int n, int k, float* a_ptr, float* b_ptr, float* c_ptr, int dev_, size_t gpu_offload=0, int mklThreads=4);

int main(int argc, char** argv) {
    printf("\n-----\nmkdir timedrun fake\n\n");
    int m=4096, n=4096, k=4096, dev_=0, mklThreads=4;
    size_t gpu_offload=0;
    if(argc<5) {
        printf("USAGE: ./matmul <Matrix Size> <Location [CPU=0, GPU=1, Hybrid=2]> <GPU_offload_Size> <Total CPU Cores[1-4]>\n");
	exit(-1);
    }

    if(argc>1) m = n = k = atoi(argv[1]);
    if(argc>2) dev_ = atoi(argv[2]);
    if(argc>3) gpu_offload = atoi(argv[3]);
    if(argc>4) mklThreads = atoi(argv[4]);

    size_t matrix1_memory_size = m*n*sizeof(float);
    size_t matrix2_memory_size = n*k*sizeof(float);
    size_t matrix3_memory_size = m*k*sizeof(float);

    /* 
     * There is no significance of the parameter to function zeroCopyPtrAlignment
     * and also of the second parameter to the function zeroCopySizeAlignment,
     * hence passing these parameters as NULL
     */
    size_t alignmentForPtr = zeroCopyPtrAlignment(NULL); 
    size_t alignedSize1 = zeroCopySizeAlignment(matrix1_memory_size, NULL);
    size_t alignedSize2 = zeroCopySizeAlignment(matrix2_memory_size, NULL);
    size_t alignedSize3 = zeroCopySizeAlignment(matrix3_memory_size, NULL);

    float* matrix_A = (float*)aligned_malloc(alignedSize1, alignmentForPtr);
    if (!matrix_A) {
	printf("Out of memory\n");
	exit(0);
    }

    float* matrix_B = (float*)aligned_malloc(alignedSize2, alignmentForPtr);
    if (!matrix_B) {
	printf("Out of memory\n");
	exit(0);
    }
	
    float* matrix_C = (float*)aligned_malloc(alignedSize3, alignmentForPtr);
    if (!matrix_C) {
	printf("Out of memory\n");
	exit(0);
    }

    for (size_t i = 0; i < m; ++i) {
        float* row_A = matrix_A + i*n;
        std::fill(row_A, row_A + n, float(1));
    }

    for (size_t i = 0; i < n; ++i) {
        float* row_B = matrix_B+ i*k;
        std::fill(row_B, row_B + k, float(1));
    }

    for (size_t i = 0; i < m; ++i) {
        float* row_C = matrix_C+ i*k;
        std::fill(row_C, row_C + k, float(0));
    }

    gemm(m, n, k, matrix_A, matrix_B, matrix_C, dev_, gpu_offload, mklThreads);
}

