#include <cmath>
#include <ctime>
#include <limits>
#include <iostream>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#include <CL/cl.h>

#include "mkl.h"
#include "timer.h"
#include "basic.hpp"
#include "oclobject.hpp"

#define CPU 0
#define GPU 1
#define HYBRID 2

float RATIO = 0.25;
const int VERIFY = 1;

struct thread_args {
    int cpu_offload;
    int gpu_offload;
    int size2;
    int size3;
    const float* matrix_A;
    const float* matrix_B;
    float* matrix_C;
    float alpha;
    float beta;
    int mklThreads;
};

void* mkl_thread(void* args) {

    struct thread_args* mkl_args = (struct thread_args*)args;

    mkl_set_num_threads(mkl_args->mklThreads);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mkl_args->cpu_offload, mkl_args->size2, mkl_args->size3, mkl_args->alpha, mkl_args->matrix_A + mkl_args->gpu_offload * mkl_args->size3 , mkl_args->size3, mkl_args->matrix_B, mkl_args->size2, mkl_args->beta, mkl_args->matrix_C + mkl_args->gpu_offload * mkl_args->size3, mkl_args->size2);
   
    END_TIMER_CPU 
}

struct OpenGEMMProgram : public OpenCLProgram {
	cl_kernel kernel;

	OpenGEMMProgram(
		OpenCLBasic& oclobjects,
		const std::wstring& program_file_name,
		const string& program_text,
		const string& kernel_name,
		const string& build_options = ""
		);

	~OpenGEMMProgram();

	cl_kernel CreateKernel(const string& kernel_name);

private:

	OpenGEMMProgram(const OpenGEMMProgram&);
	OpenGEMMProgram& operator= (const OpenGEMMProgram&);
};

OpenGEMMProgram::OpenGEMMProgram(
	OpenCLBasic& oclobjects,
	const std::wstring& program_file_name,
	const string& program_text,
	const string& kernel_name,
	const string& build_options
	) :
	OpenCLProgram(oclobjects, program_file_name, program_text, build_options),
	kernel(0) {}

OpenGEMMProgram::~OpenGEMMProgram() {
	try {
		if (kernel) {
			clReleaseKernel(kernel);
		}
	}
	catch (...) {
		destructorException();
	}
}

cl_kernel OpenGEMMProgram::CreateKernel(const string& kernel_name) {
	using namespace std;

	cl_int err = 0;
	kernel = clCreateKernel(program, kernel_name.c_str(), &err);
	SAMPLE_CHECK_ERRORS(err);
}

void gemm(int m, int n, int k, float* a_ptr, float* b_ptr, float* c_ptr, int dev_, size_t gpu_offload=0, int mklThreads=4) {
            if (dev_ == CPU || !gpu_offload) {
                RATIO = 0.0;
		dev_ = CPU;
            }
            else if(dev_ == GPU || gpu_offload==m) {
                RATIO = 1.0;
		gpu_offload=0;
		dev_ = GPU;
            }
               
            OpenCLBasic oclobjects(
                "Intel",
                "gpu",
                "0"/*, 
                CL_QUEUE_PROFILING_ENABLE  */
            );
            
            string build_option = " -cl-mad-enable -DGPU";
            OpenGEMMProgram executable(oclobjects, L"/home/vivekk/tf-ocl/standalone/matmul.cl", "", "",
                    build_option);
            cl_int err = 0;
            cl_mem dev_A, dev_B, dev_C;

            if(!gpu_offload) gpu_offload = m * RATIO;
            size_t cpu_offload = m - gpu_offload;

            printf("GPU offload = %lu\n", gpu_offload);
            printf("CPU offload = %lu\n", cpu_offload);
	    PCM_INIT
            START_TIMER
	    PCM_START
            if(dev_ == GPU || dev_ == HYBRID) {
                
                dev_A = clCreateBuffer(
                            oclobjects.context,
                            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            gpu_offload * k * sizeof(float),
                            (void*)a_ptr,
                            &err
                        );
                SAMPLE_CHECK_ERRORS(err);

                dev_B = clCreateBuffer(
                            oclobjects.context,
                            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            k * n * sizeof(float),
                            (void*)b_ptr,
                            &err
                        );
                SAMPLE_CHECK_ERRORS(err);

                dev_C = clCreateBuffer(
                            oclobjects.context,
                            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            gpu_offload * n * sizeof(float),
                            (void*)c_ptr,
                            &err
                        );
                SAMPLE_CHECK_ERRORS(err);

            }
            float alpha = 1;
            float beta = 0;
            
            pthread_t thread;
           
            if(dev_ == CPU) {
		START_TIMER_CPU
                mkl_set_num_threads(mklThreads);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        cpu_offload, k, n, alpha, a_ptr + gpu_offload * n , n,
                        b_ptr, k, beta, c_ptr + gpu_offload * n, k);
		END_TIMER_CPU
            }

            if (dev_ == HYBRID) {
		START_TIMER_CPU;
                struct thread_args* mkl_args;
                mkl_args = (struct thread_args*)malloc(sizeof(
                            struct thread_args));
                mkl_args->cpu_offload = cpu_offload;
                mkl_args->gpu_offload = gpu_offload;
                mkl_args->size2 = k;
                mkl_args->size3 = n;
                mkl_args->matrix_A = a_ptr;
                mkl_args->matrix_B = b_ptr;
                mkl_args->matrix_C = c_ptr;
                mkl_args->alpha = alpha;
                mkl_args->beta = beta;
                mkl_args->mklThreads= mklThreads;

                pthread_create(&thread, NULL, mkl_thread, (void*)mkl_args);
            }

            if (dev_ == GPU || dev_ == HYBRID) {
		START_TIMER_GPU;
                executable.CreateKernel("L3_SLM_8x8_8x16");
                err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem),
                        &dev_A);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem),
                        &dev_B);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 2, sizeof(cl_mem),
                        &dev_C);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 3, sizeof(float),
                        &alpha);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 4, sizeof(float),
                        &beta);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 5, sizeof(int),
                        &k);
                SAMPLE_CHECK_ERRORS(err);
                err = clSetKernelArg(executable.kernel, 6, sizeof(int),
                        &n);
                SAMPLE_CHECK_ERRORS(err);

                size_t block_x = 8;
                size_t block_y = 8;
                size_t group_x = 16;
                size_t group_y = 8;

                size_t global_size[2] = { n/block_x , gpu_offload/block_y };
                size_t local_size[2] = { group_x, group_y };

                cl_event cl_perf_event = NULL;

                err = clEnqueueNDRangeKernel(
                        oclobjects.queue,
                        executable.kernel,
                        2,
                        0,
                        global_size,
                        local_size,
                        0, 0, &cl_perf_event
                    );
                SAMPLE_CHECK_ERRORS(err);

                err = clWaitForEvents(1, &cl_perf_event);
                SAMPLE_CHECK_ERRORS(err);
		END_TIMER_GPU
            }

            if (dev_ == HYBRID) {
                pthread_join(thread, NULL);
            }
	    PCM_STOP
	    END_TIMER
	    PCM_FINALIZE
 
            if (VERIFY == 1) {

                for(size_t i = 0; i < m * n; i++) {
                    if (c_ptr[i] != m) {
                        printf("Error, index %d: Wanted %f, got %f\n", i,
                                m, c_ptr[i]);
                        break;
                    }
                }
            }
	    PRINT_HARNESS_FOOTER
}

