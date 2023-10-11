// #pragma once
#include "util.h"
class testCase{
public:
    int inside;
    int chn;
    int bat4Conv;
    int nInputTran, nKernelTran;
    float *inputTran_cpu;
    float *kernelTran_cpu;
    float *inputTran_gpu;  //, *output_gpu;
    float *kernelTran_gpu;

    char inputTranName[60];
    char kernelTranName[60];

    testCase(char* fileName1, char* fileName2, int M, int N, int K){
        strcpy(inputTranName, fileName1);
        strcpy(kernelTranName, fileName2);
        inside = size;
        // bat4Conv = batch;
        chn = channel;
        
        nInputTran = M*K*36;
        nKernelTran = N*K*36;
        inputTran_cpu = get_parameter(inputTranName, nInputTran);
        kernelTran_cpu = get_parameter(kernelTranName, nKernelTran);
        
        cudaMalloc((void **) &inputTran_gpu, nInput<<2);
        cudaMalloc((void **) &kernelTran_gpu, nInput<<2);
        
        cudaMemcpy(inputTran_gpu, inputTran_cpu, nInputTran<<2, cudaMemcpyHostToDevice);
        cudaMemcpy(kernelTran_gpu, kernelTran_cpu, nKernelTran<<2, cudaMemcpyHostToDevice);
    }
};