#pragma once
#include <bits/stdc++.h>
#include <cudnn.h>
#include "wrapedConv_NCHW.cuh"
#include "wrapedConv_NHWC.cuh"
#include "wrapedConv_CHWN.cuh"


void convLayoutTrans_CHWN_to_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
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

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaMalloc((void **) &workSpace,  nWorkSpace*sizeof(float));

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

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

    // input: CHWN layout
    // output: KM matrix,  M=(blockn, blockn, batch)
    wino_input_trans_chwn_suitFor128<<< dim3(blockn, blockn, chn), dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );  
    // cudaEventRecord(ts1, NULL);

    // input: CHWN layout
    // output: KN matrix
    wino_kernel_trans_chwn_suitFor128<<< dim3(chn,1,1), dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);

    // input: KMKN matrix,  M=(blockn, blockn, batch)
    // output: MN matrix,  M=(blockn, blockn, batch)
    GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize,MSize,KSize,1, filterTran_gpu, inputTran_gpu,0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);
    
    // input: MN matrix,  M=(blockn, blockn, batch)
    // output: NCHW layout
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
    // cudaFree(workSpace);
}

void convLayoutTrans_CHWN_to_NHWC(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
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

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaMalloc((void **) &workSpace,  nWorkSpace*sizeof(float));

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

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
    // input: CHWN layout
    // output: KM matrix,  M=(blockn, blockn, batch)
    wino_input_trans_chwn_suitFor128<<< dim3(blockn, blockn, chn), dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );  
    // cudaEventRecord(ts1, NULL);

    // input: CHWN layout
    // output: KN matrix
    wino_kernel_trans_chwn_suitFor128<<< dim3(chn,1,1), dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);

    // input: KMKN matrix,  M=(blockn, blockn, batch)
    // output: MN matrix,  M=(blockn, blockn, batch)
    // change the sequence of two matrix of input and kernel.
    GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize,MSize,KSize,1, filterTran_gpu, inputTran_gpu,0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);
    
    // input: MN matrix,  M=(blockx, blocky, batch)
    // output: NHWC output
    // wino_invers_nchw_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    wino_invers_nhwc_suitFor128<<< dim3(bat4Conv, blockn, blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
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
    // cudaFree(workSpace);
}

void convLayoutTrans_NHWC_to_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
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

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaDeviceReset();
    // size_t free_memory, total_memory;
    // cudaMemGetInfo(&free_memory, &total_memory);
    // printf("可用GPU内存: %zu MB\n", free_memory / (1024 * 1024));
    // printf("总GPU内存: %zu MB\n", total_memory / (1024 * 1024));
    // printf("请求的内存: %zu MB\n", nWorkSpace * sizeof(float) / (1024 * 1024));
    // nWorkSpace = 536870912;
    // if (nWorkSpace * sizeof(float) > free_memory) {
    //     printf("请求的内存超过可用内存，尝试减少批量大小或输入尺寸\n");
    //     return;
    // }
    // cudaDeviceSynchronize();
    // CHECK_CUDA(cudaMalloc((void **) &workSpace,  nWorkSpace<<2));
    // cudaMalloc((void **) &workSpace,  nWorkSpace<<2);
    // // printf("Size of WorkSpace:%d\n",nWorkSpace);

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran;

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
    // int nInputTran = 36*MSize*KSize;
    // int nFilterTran = 36*MSize*KSize;

    // printf("blockn, bat4Conv, chn = %d %d %d\n inside, inside_beta, MSize, KSize = %d,%d,%d,%d\n",blockn, bat4Conv, chn, inside, inside_beta,MSize,KSize);

    // input: NHWC layout
    // output: MK matrix,  M=(blockn, blockn, batch)
    // wino_input_trans_nhwc_suitFor128<<< dim3(1, 1, 1), dim3(1,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    wino_input_trans_nhwc_suitFor128<<< dim3(blockn, blockn, bat4Conv), dim3(chn,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    // cudaEventRecord(ts1, NULL);
    // float *output2 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output2, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // printf("2st:%f\n",output2[0]);

    // input: NHWC layout
    // output: NK matrix
    wino_kernel_trans_nhwc_suitFor128<<< dim3(numOfFilter,1,1), dim3(chn,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);
    // wino_kernel_trans_nhwc_suitFor128<<< dim3(1,1,1), dim3(1,1,1)>>>(NSize, KSize, m2, filterTran_gpu); 
    // float *output3 =(float *)malloc(nFilterTran * sizeof(float));
    // cudaMemcpy(output3, filterTran_gpu, nFilterTran<<2, cudaMemcpyDeviceToHost);
    // printf("3st:%f\n",output3[0]);
    
    // input: MKNK matrix,  M=(blockn, blockn, batch)
    // output: MN matrix,  M=(blockn, blockn, batch)
    // GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    //TODO:
    GEMM_batch_256_128x128_MKNK<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize, NSize, KSize, 1, inputTran_gpu, filterTran_gpu, 0, gemmOutput_gpu);

    // input: MN matrix, M=(blockn, blockn, batch)
    // output NCHW layout.
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
    // cudaFree(workSpace);
}

void convLayoutTrans_NHWC_to_CHWN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
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

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaDeviceReset();
    // size_t free_memory, total_memory;
    // cudaMemGetInfo(&free_memory, &total_memory);
    // printf("可用GPU内存: %zu MB\n", free_memory / (1024 * 1024));
    // printf("总GPU内存: %zu MB\n", total_memory / (1024 * 1024));
    // printf("请求的内存: %zu MB\n", nWorkSpace * sizeof(float) / (1024 * 1024));
    // nWorkSpace = 536870912;
    // if (nWorkSpace * sizeof(float) > free_memory) {
    //     printf("请求的内存超过可用内存，尝试减少批量大小或输入尺寸\n");
    //     return;
    // }
    // cudaDeviceSynchronize();
    // CHECK_CUDA(cudaMalloc((void **) &workSpace,  nWorkSpace<<2));
    // cudaMalloc((void **) &workSpace,  nWorkSpace<<2);
    // // printf("Size of WorkSpace:%d\n",nWorkSpace);

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran;

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
    // int nInputTran = 36*MSize*KSize;
    // int nFilterTran = 36*MSize*KSize;

    // printf("blockn, bat4Conv, chn = %d %d %d\n inside, inside_beta, MSize, KSize = %d,%d,%d,%d\n",blockn, bat4Conv, chn, inside, inside_beta,MSize,KSize);

    // input: NHWC layout
    // output: MK matrix,  M=(blockn, blockn, batch)
    // wino_input_trans_nhwc_suitFor128<<< dim3(1, 1, 1), dim3(1,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    wino_input_trans_nhwc_suitFor128<<< dim3(blockn, blockn, bat4Conv), dim3(chn,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    // cudaEventRecord(ts1, NULL);
    // float *output2 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output2, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // printf("2st:%f\n",output2[0]);

    // input: NHWC layout
    // output: NK matrix
    wino_kernel_trans_nhwc_suitFor128<<< dim3(numOfFilter,1,1), dim3(chn,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    // input: MKNK matrix,  M=(blockn, blockn, batch)
    // output: NM matrix,  M=(blockn, blockn, batch)
    // GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    GEMM_batch_256_128x128_MKNK<<<dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize, MSize, KSize, 1, filterTran_gpu,inputTran_gpu, 0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>> (NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);

    // cudaDeviceSynchronize();
    // sumVar<<<1,1>>>( 36, MSize, NSize, gemmOutput_gpu);
    // input: NM matrix,  M=(blockn, blockn, batch)
    // output: CHWN layout
    wino_invers_chwn_suitFor128_2<<< dim3(numOfFilter,blockn,blockn), dim3(bat4Conv,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_chwn_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // cudaEventRecord(ts4, NULL);
    
    // cudaDeviceSynchronize();
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu1 =(float *)malloc(nConvOutput * sizeof(float));
    // cudaMemcpy(convOutput_gpu1, *output, nConvOutput<<2, cudaMemcpyDeviceToHost);
    // int cnt4 = save_parameter("result/conv_CHWN.bin", nConvOutput, convOutput_gpu1);
    // free(convOutput_gpu1);

    chn = numOfFilter;

    

    cudaDeviceSynchronize();
    // cudaFree(workSpace);
}

void convLayoutTrans_NCHW_to_NHWC(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
    int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);

    int blockn = (inside_beta -2) /4;
    
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;
    // int K = chn;

    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    //So strange here, why all of this work when I add this "+1", need to figure out !!!
    int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaMalloc((void **) &workSpace,  nWorkSpace<<2);

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;

    // input: NCHW layout
    // output: KM matrix,  M=(blockx, blocky, batch)
    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );

    // input: NCHW layout
    // output: KN matrix
    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    // input: KNKM matrix,  M=(blockn, blockn, batch)
    // output: MN matrix,  M=(blockn, blockn, batch)
    GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);

    // input: MN matrix,  M=(blockx, blocky, batch)
    // output: NHWC output
    // wino_invers_nchw_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    wino_invers_nhwc_suitFor128<<< dim3(bat4Conv, blockn, blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
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
    // cudaFree(workSpace);
}

void convLayoutTrans_NCHW_to_CHWN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
        int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);

    int blockn = (inside_beta -2) /4;
    
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;
    // int K = chn;

    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    //So strange here, why all of this work when I add this "+1", need to figure out !!!
    int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;

    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;
    // int nGemmOutput = 36 * MSize*NSize;

    // float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    // cudaMalloc((void **) &workSpace,  nWorkSpace<<2);

    // inputTran_gpu = workSpace;
    // filterTran_gpu = workSpace + nInputTran;
    // gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;

    // input: NCHW layout
    // output: KM matrix,  M=(blockx, blocky, batch)
    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );

    // input: NCHW layout
    // output: KN matrix
    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    // input: KNKM matrix,  M=(blockn, blockn, batch)
    // output: NM matrix,  M=(blockn, blockn, batch)
    GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);

    // input: NM matrix,  M=(blockn, blockn, batch)
    // output: CHWN layout
    // wino_invers_nchw_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    wino_invers_chwn_suitFor128_2<<< dim3(numOfFilter,blockn,blockn), dim3(bat4Conv,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
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
    // cudaFree(workSpace);
}

// 定义数据布局枚举类型
enum class DataLayout {
    NCHW,
    NHWC, 
    CHWN
};

// 顶层管理函数
void layoutManager(
    DataLayout srcLayout,
    DataLayout dstLayout,
    int bat4Conv, 
    int inside,
    int& chn,
    int numOfFilter,
    int padding,
    float *input,
    float *weights,
    float *inputTran_gpu,
    float *filterTran_gpu,
    float *gemmOutput_gpu,
    float **output
) {
    if(srcLayout == dstLayout) {
        // 相同布局直接调用对应的卷积函数
        switch(srcLayout) {
            case DataLayout::NCHW:
                wrapedConv_NCHW(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
                break;
            case DataLayout::NHWC:
                wrapedConv_NHWC(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
                break;
            case DataLayout::CHWN:
                wrapedConv_CHWN(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
                break;
        }
        return;
    }

    // 不同布局间的转换
    switch(srcLayout) {
        case DataLayout::CHWN:
            if(dstLayout == DataLayout::NCHW) {
                convLayoutTrans_CHWN_to_NCHW(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            } else {
                convLayoutTrans_CHWN_to_NHWC(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            }
            break;
            
        case DataLayout::NCHW:
            if(dstLayout == DataLayout::CHWN) {
                convLayoutTrans_NCHW_to_CHWN(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            } else {
                convLayoutTrans_NCHW_to_NHWC(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            }
            break;
            
        case DataLayout::NHWC:
            if(dstLayout == DataLayout::NCHW) {
                convLayoutTrans_NHWC_to_NCHW(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            } else {
                convLayoutTrans_NHWC_to_CHWN(bat4Conv, inside, chn, numOfFilter, padding, input, weights, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, output);
            }
            break;
    }
}