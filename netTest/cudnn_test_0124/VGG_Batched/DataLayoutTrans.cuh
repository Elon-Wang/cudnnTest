#pragma once
#include <bits/stdc++.h>
#include <cudnn.h>
#include "wrapedConv_NCHW.cuh"
#include "wrapedConv_NHWC.cuh"
#include "wrapedConv_CHWN.cuh"

// void convLayoutTrans(cudnnTensorFormat_t inFormat, cudnnTensorFormat_t outFormat){
//     if (inFormat ==-1 && outFormat){}
// }


void convLayoutTrans(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
    // CHWN to NCHW

    int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);
    
    int blockn = (inside_beta -2) /4;
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;

    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    int nInputTran = 36 * MSize * KSize;
    int nFilterTran = 36 * NSize * KSize;
    int nGemmOutput = 36 * MSize*NSize;

    float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    cudaMalloc((void **) &workSpace,  nWorkSpace*sizeof(float));

    inputTran_gpu = workSpace;
    filterTran_gpu = workSpace + nInputTran;
    gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

    int blockx, blocky, bat4Gemm;

    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm =36;
    
    int oside = inside + 2*padding -2;

    // cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    // cudaEventCreate(&ts0);
    // cudaEventCreate(&ts1);
    // cudaEventCreate(&ts2);
    // cudaEventCreate(&ts3);
    // cudaEventCreate(&ts4);

    // cudaEventRecord(ts0, NULL);

    wino_input_trans_chwn_suitFor128<<< dim3(blockn, blockn, chn), dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );  
    // cudaEventRecord(ts1, NULL);

    wino_kernel_trans_chwn_suitFor128<<< dim3(chn,1,1), dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);

    // change the sequence of two matrix of input and kernel.
    GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize,MSize,KSize,1, filterTran_gpu, inputTran_gpu,0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);
    

    wino_invers_nchw_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_chwn_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // cudaEventRecord(ts4, NULL);

    chn = numOfFilter;

    

    cudaDeviceSynchronize();
    // float t0, t1, t2, t3, total;
    // cudaEventElapsedTime(&t0, ts0, ts1);
    // cudaEventElapsedTime(&t1, ts1, ts2);
    // cudaEventElapsedTime(&t2, ts2, ts3);
    // cudaEventElapsedTime(&t3, ts3, ts4);
    // cudaEventDestroy(ts0);
    // cudaEventDestroy(ts1);
    // cudaEventDestroy(ts2);
    // cudaEventDestroy(ts3);
    // cudaEventDestroy(ts4);
    // total = t0+t1+t2+t3;

    // printf("time:%lf ms\t (%f, %f, %f, %f)\n", (total),(100*t0/total),(100*t1/total),(100*t2/total),(100*t3/total) );
    cudaFree(workSpace);
}