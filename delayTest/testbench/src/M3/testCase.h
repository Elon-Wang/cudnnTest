// #pragma once
#include "util.h"
class testCase{
public:
    int inside_beta;
    int chn;
    int bat4Conv;
    int blockn;
    int nInputTran, nKernelTran;
    int M, N, K;
    int MSize, NSize, KSize;
    float *inputTran_cpu;
    float *kernelTran_cpu;
    float *inputTran_gpu;  //, *output_gpu;
    float *kernelTran_gpu;

    char inputTranName[60];
    char kernelTranName[60];

    testCase(char* fileName1, char* fileName2,  int size, int batch, int channel, int channel_NextLevel ){
        bat4Conv = batch;
        chn = channel;
        int padding =1;
        int marginOfInputSide = (size+2*padding-6)%4;
        bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
        inside_beta = sideCheck? size +2*padding : (size + 2*padding + 4 - marginOfInputSide);
        blockn = (inside_beta -2) /4;
        M = bat4Conv * blockn * blockn;
        N = channel_NextLevel;
        K = channel;

        strcpy(inputTranName, fileName1);
        strcpy(kernelTranName, fileName2);

        bool MCheck = ( M % 128 == 0 )? true: false;
        bool NCheck = ( N % 128 == 0 )? true: false;
        bool KCheck = ( K % 8 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        NSize = NCheck? N: ((N/128 +1) * 128);
        KSize = KCheck? K: ((K/8 +1) * 8);

        nInputTran = 36* MSize *KSize;
        nKernelTran = 36* NSize *KSize;
        inputTran_cpu = get_parameter(inputTranName, nInputTran);
        kernelTran_cpu = get_parameter(kernelTranName, nKernelTran);
        
        cudaMalloc((void **) &inputTran_gpu, nInputTran<<2);
        cudaMalloc((void **) &kernelTran_gpu, nKernelTran<<2);
        
        cudaMemcpy(inputTran_gpu, inputTran_cpu, nInputTran<<2, cudaMemcpyHostToDevice);
        cudaMemcpy(kernelTran_gpu, kernelTran_cpu, nKernelTran<<2, cudaMemcpyHostToDevice);
    }
};