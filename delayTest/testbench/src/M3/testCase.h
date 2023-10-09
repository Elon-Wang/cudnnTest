// #pragma once
#include "util.h"
class testCase{
public:
    int inside;
    int chn;
    int bat4Conv;
    int nInput;
    float *input_cpu;
    float *kernel_cpu;
    float *input_gpu;  //, *output_gpu;
    float *kernel_gpu;

    char inputname[60];
    char kernelname[60];

    testCase(char* fileName1, char* fileName2, int M, int N, int K, int batch){
        strcpy(inputname, fileName1);
        strcpy(kernelname, fileName2);
        inside = size;
        bat4Conv = batch;
        chn = channel;
        
        nInput = M*K*batch;
        nKernel = N*K*batch;
        input_cpu = get_parameter(inputname, nInput);
        kernel_cpu = get_parameter(kernelname, nKernel);
        
        cudaMalloc((void **) &input_gpu, nInput<<2);
        cudaMalloc((void **) &kernel_gpu, nInput<<2);
        
        cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_gpu, kernel_cpu, nInput<<2, cudaMemcpyHostToDevice);
    }
};