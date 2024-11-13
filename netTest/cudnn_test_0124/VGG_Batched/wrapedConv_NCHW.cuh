#pragma once
#include "GEMM.cuh"

#ifndef WARMUP
#define WARMUP
__global__ void warmup(){}
#endif
// void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float **output);
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float **output);
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs, int bound);
__global__ void wino_input_trans_nchw_suitFor128_2(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs, int bound, int tileArray, int numOfBlcokn);
__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);
__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);
__global__ void wino_invers_nchw_suitFor128_2(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);
__global__ void wino_invers_nchw_suitFor128_3(int oside, int MSize, int NSize, float* pInputs, float* pOutputs, int tileArray, int numOfBlcokn);

/*  // wrapedConv_NCHW original__version
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
// void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
 
    // int padding =1;
    // int size = numOfFilter * chn * (inside-2+2*padding) * (inside-2+2*padding) ;
    
    // take care of the resize code
    // if (*output != NULL) {
    //     cudaFree(*output);
    // }
    // assert(inside >=4);

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
    // int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;
    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;

    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);


    //TBC, gemm part
    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;
    // float *inputTran_gpu, *filterTran_gpu;
    // float *gemmOutput_gpu;

    // cudaMalloc(output, size *sizeof(float));
    // cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    // cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    // cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);

    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);
    cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    cudaEventCreate(&ts0);
    cudaEventCreate(&ts1);
    cudaEventCreate(&ts2);
    cudaEventCreate(&ts3);
    cudaEventCreate(&ts4);

    cudaEventRecord(ts0, NULL);
    

    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );

    cudaEventRecord(ts1, NULL);

    // float *output1 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output1, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // const char module1Name[] = "../data/output1.bin";
    // int cnt4 = save_parameter(module1Name, nInputTran, output1);

    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    cudaEventRecord(ts2, NULL);

    GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);

    cudaEventRecord(ts3, NULL);

    // wino_invers_chwn_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);
    wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);

    cudaEventRecord(ts4, NULL);
    // printf("blockn:%d\n",blockn);

    chn = numOfFilter;

    // cudaMemcpy(output, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    float t0, t1, t2, t3, total;
    cudaEventElapsedTime(&t0, ts0, ts1);
    cudaEventElapsedTime(&t1, ts1, ts2);
    cudaEventElapsedTime(&t2, ts2, ts3);
    cudaEventElapsedTime(&t3, ts3, ts4);
    cudaEventDestroy(ts0);
    cudaEventDestroy(ts1);
    cudaEventDestroy(ts2);
    cudaEventDestroy(ts3);
    cudaEventDestroy(ts4);
    total = t0+t1+t2+t3;

    printf("time:%lf ms\t (%f, %f, %f, %f)\n", (total),(100*t0/total),(100*t1/total),(100*t2/total),(100*t3/total) );

    // cudaFree(inputTran_gpu);
    // cudaFree(filterTran_gpu);
    // cudaFree(gemmOutput_gpu);
    // printf("m1 before free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(*m1);
    // *m1 = nullptr;
    // printf("m1 after free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(convOutput_gpu);
}
*/

/* //wrapedConv_NCHW_2
// Change the M from (blockn, blockn, batch) to (batch, blockn, blockn)
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
// void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
 
    // int padding =1;
    // int size = numOfFilter * chn * (inside-2+2*padding) * (inside-2+2*padding) ;
    
    // take care of the resize code
    // if (*output != NULL) {
    //     cudaFree(*output);
    // }
    // assert(inside >=4);

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
    // int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;
    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;

    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);


    //TBC, gemm part
    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;
    // float *inputTran_gpu, *filterTran_gpu;
    // float *gemmOutput_gpu;

    // cudaMalloc(output, size *sizeof(float));
    // cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    // cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    // cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);

    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);
    // cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    // cudaEventCreate(&ts0);
    // cudaEventCreate(&ts1);
    // cudaEventCreate(&ts2);
    // cudaEventCreate(&ts3);
    // cudaEventCreate(&ts4);

    // cudaEventRecord(ts0, NULL);
    

    // wino_input_trans_nchw_suitFor128_2<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );
    
    printf("blockn:%d\n",blockn);
    if (blockn<=14){
        // wino_input_trans_nchw_suitFor128<<<dim3(bat4Conv,chn,1),dim3(blockn,blockn,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput);
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn,1),dim3(blockn,blockn,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput, 1, blockn);
    } else{
        int tileArray = ((blockn+13)/14 ) ;
        // wino_input_trans_nchw_suitFor128<<<dim3(bat4Conv,chn,1),dim3(blockn,blockn,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput);
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn, tileArray*tileArray),dim3(14,14,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu, nInput, tileArray, blockn);
    }

    // printf("blockn:%d\n",blockn);
    // wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn,1),dim3(blockn,14,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput,1,blockn);    
    // wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );

    // cudaEventRecord(ts1, NULL);

    // float *output1 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output1, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // const char module1Name[] = "../data/output1.bin";
    // int cnt4 = save_parameter(module1Name, nInputTran, output1);

    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    // cudaEventRecord(ts2, NULL);

    GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);

    // cudaEventRecord(ts3, NULL);

    // wino_invers_chwn_suitFor128_2<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);
    // wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    wino_invers_nchw_suitFor128_2<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);

    // cudaEventRecord(ts4, NULL);
    // printf("blockn:%d\n",blockn);

    chn = numOfFilter;

    // cudaMemcpy(output, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToDevice);
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

    // cudaFree(inputTran_gpu);
    // cudaFree(filterTran_gpu);
    // cudaFree(gemmOutput_gpu);
    // printf("m1 before free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(*m1);
    // *m1 = nullptr;
    // printf("m1 after free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(convOutput_gpu);
}
*/

/*
__global__ void debugDisplay( int MSize, int NSize, float* gemmOutput_gpu){
    int size = MSize *NSize;
    printf("MSize:%d NSize:%d, Size:%d\n gemmOuptut:\n",MSize, NSize, size);
    for (int i=0;i<6;i++){
        for (int j=0;j<6;j++){
            int idx = 6*i*size + j*size;
            printf("%.3f \t",gemmOutput_gpu[idx]);
        }printf("\n");
    }
}

__global__ void tranMatrixDisplay( int KSize, int mainSize,  float* tranMatrix){
    printf("Array KSize:%d MainSize:%d\n",KSize,mainSize);
    for(int i=0;i<KSize;i++){
        printf("%.3f  ",tranMatrix[i*mainSize]);
    }printf("\n");
}

__global__ void sumLayer(int bat4Conv,int chn, int inside, float *featureMap){
    float sum=0;
    for (int c=0;c<chn;c++){
        for (int i=0;i<inside;i++){
            for (int j=0;j<inside;j++){
                sum += featureMap[c*224*224+i*224+j];
            }
        }
    }
    printf("the sum of the feature map:%f\n",sum);
}
*/

/* // wrapedConv_NCHW_3
// exchange the order of filterTran and inputTran given to the gemm
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
 
    // int padding =1;
    // int size = numOfFilter * chn * (inside-2+2*padding) * (inside-2+2*padding) ;
    
    // take care of the resize code
    // if (*output != NULL) {
    //     cudaFree(*output);
    // }
    // assert(inside >=4);

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
    // int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;
    // int nInputTran = 36 * MSize * KSize;
    // int nFilterTran = 36 * NSize * KSize;

    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);


    //TBC, gemm part
    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;
    // float *inputTran_gpu, *filterTran_gpu;
    // float *gemmOutput_gpu;

    // cudaMalloc(output, size *sizeof(float));
    // cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    // cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    // cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);

    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);
    

    // wino_input_trans_nchw_suitFor128_2<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );
    
    if (blockn<=14){
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn,1),dim3(blockn,blockn,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput, 1, blockn);
    } else{
        int tileArray = ((blockn+13)/14 ) ;
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn, tileArray*tileArray),dim3(14,14,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu, nInput, tileArray, blockn);
    }
    
    // tranMatrixDisplay<<<1,1>>>(KSize,MSize,inputTran_gpu);


    // float *output1 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output1, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // const char module1Name[] = "../data/output1.bin";
    // int cnt4 = save_parameter(module1Name, nInputTran, output1);

    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // tranMatrixDisplay<<<1,1>>>(KSize,NSize,filterTran_gpu);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu, 0, gemmOutput_gpu);
    // debugDisplay<<<1,1>>>(MSize,NSize,gemmOutput_gpu);
    // wino_invers_nchw_suitFor128_2<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // cudaDeviceSynchronize();

    // printf("NSize :%d \tMSize:%d \tKSize:%d \t\n",NSize, MSize,KSize);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu, 0, gemmOutput_gpu);
    GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>> (NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // cudaDeviceSynchronize();

    // int nGemmOutput = 36 * MSize*NSize;
    // float *output1 =(float *)malloc(nGemmOutput * sizeof(float));
    // cudaMemcpy(output1, gemmOutput_gpu, nGemmOutput<<2, cudaMemcpyDeviceToHost);
    // int cnt3 = save_parameter("result/gemm2.bin", nGemmOutput, output1);
    // free(output1);
    // cudaDeviceSynchronize();
    // debugDisplay<<<1,1>>>(MSize,NSize,gemmOutput_gpu);
    // wino_invers_chwn_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);

    // wino_invers_nchw_suitFor128_2<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    if (blockn<=14){
        wino_invers_nchw_suitFor128_3<<<dim3(bat4Conv, numOfFilter, 1), dim3(blockn,blockn,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, 1, blockn);
    } else{
        int tileArray = ((blockn+13)/14 ) ;
        // printf("tileArray:%d\n",tileArray);
        wino_invers_nchw_suitFor128_3<<<dim3(bat4Conv, numOfFilter, tileArray*tileArray), dim3(14,14,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, tileArray, blockn);
    }
    // wino_invers_nchw_suitFor128_2<<<dim3(1, 1, 1), dim3(1,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_nchw_suitFor128_2<<<dim3(1, 1, 1), dim3(1,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, 1, blockn);
    // int nGemmOutput = 36 * MSize*NSize;
    // float *output1 =(float *)malloc(nGemmOutput * sizeof(float));
    // cudaMemcpy(output1, *output, nGemmOutput<<2, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu1 =(float *)malloc(nConvOutput * sizeof(float));
    // cudaMemcpy(convOutput_gpu1, *output, nConvOutput<<2, cudaMemcpyDeviceToHost);
    // int cnt4 = save_parameter("result/conv_out2.bin", nConvOutput, convOutput_gpu1);
    // free(convOutput_gpu1);
    // cudaDeviceSynchronize();
    // sumLayer<<<1,1>>>( bat4Conv, chn, inside, *output);

    // printf("blockn:%d\n",blockn);
    // int nGemmOutput = 36 * MSize*NSize;
    // float *gemmOutput_gpu1;
    // cudaMalloc((void **) &gemmOutput_gpu1, nGemmOutput<<2);

    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu1;
    // cudaMalloc((void **) &convOutput_gpu1, nConvOutput<<2);
    // cudaDeviceSynchronize();

    

    chn = numOfFilter;

    // cudaMemcpy(output, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // cudaFree(inputTran_gpu);
    // cudaFree(filterTran_gpu);
    // cudaFree(gemmOutput_gpu);
    // printf("m1 before free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(*m1);
    // *m1 = nullptr;
    // printf("m1 after free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(convOutput_gpu);
}
*/

// wrapedConv_NCHW_4
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
 
    // int padding =1;
    // int size = numOfFilter * chn * (inside-2+2*padding) * (inside-2+2*padding) ;
    
    // take care of the resize code
    // if (*output != NULL) {
    //     cudaFree(*output);
    // }
    // assert(inside >=4);

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

    int nInputTran = 36 * MSize * KSize;
    int nFilterTran = 36 * NSize * KSize;
    int nGemmOutput = 36 * MSize*NSize;

    float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    cudaMalloc((void **) &workSpace,  nWorkSpace<<2);

    inputTran_gpu = workSpace;
    filterTran_gpu = workSpace + nInputTran;
    gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

    


    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);


    //TBC, gemm part
    int blocky, blockx, bat4Gemm;
    // int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    // nGemmOutput = 36 * MSize*NSize;
    int oside = inside+2*padding - 2;
    // float *inputTran_gpu, *filterTran_gpu;
    // float *gemmOutput_gpu;

    // cudaMalloc(output, size *sizeof(float));
    // cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    // cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    // cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);

    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);
    // cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    // cudaEventCreate(&ts0);
    // cudaEventCreate(&ts1);
    // cudaEventCreate(&ts2);
    // cudaEventCreate(&ts3);
    // cudaEventCreate(&ts4);

    // cudaEventRecord(ts0, NULL);
    

    // wino_input_trans_nchw_suitFor128_2<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput );
    
    if (blockn<=14){
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn,1),dim3(blockn,blockn,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu,nInput, 1, blockn);
    } else{
        int tileArray = ((blockn+13)/14 ) ;
        wino_input_trans_nchw_suitFor128_2<<<dim3(bat4Conv,chn, tileArray*tileArray),dim3(14,14,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu, nInput, tileArray, blockn);
    }
    

    // cudaEventRecord(ts1, NULL);

    // float *output1 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output1, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // const char module1Name[] = "../data/output1.bin";
    // int cnt4 = save_parameter(module1Name, nInputTran, output1);

    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    // cudaEventRecord(ts2, NULL);

    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu, 0, gemmOutput_gpu);
    GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>> (NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);

    // cudaEventRecord(ts3, NULL);

    // wino_invers_chwn_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);

    // wino_invers_nchw_suitFor128_2<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    if (blockn<=14){
        wino_invers_nchw_suitFor128_3<<<dim3(bat4Conv, numOfFilter, 1), dim3(blockn,blockn,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, 1, blockn);
    } else{
        int tileArray = ((blockn+13)/14 ) ;
        wino_invers_nchw_suitFor128_3<<<dim3(bat4Conv, numOfFilter, tileArray*tileArray), dim3(14,14,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, tileArray, blockn);
    }
    // wino_invers_nchw_suitFor128_2<<<dim3(1, 1, 1), dim3(1,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_nchw_suitFor128_2<<<dim3(1, 1, 1), dim3(1,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output, 1, blockn);

    // cudaEventRecord(ts4, NULL);
    // printf("blockn:%d\n",blockn);

    chn = numOfFilter;

    // cudaMemcpy(output, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(workSpace);
    
    // int nOut = bat4Conv * inside * inside * numOfFilter;
    // float *convOutput_cpu = (float *)malloc(nOut * sizeof(float));
    // cudaMemcpy(convOutput_cpu, *output, nOut<<2, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // printf("convOutput_cpu[0]:%lf\n",convOutput_cpu[0]);
    // if ( convOutput_cpu[0] == 0.0f) {
    //     //printf("Test3:\n 1:%f\t 2:%f\n",output_cpu[0], output_cpu[1]);
    //     printf("moduleConv error\n");
    // }
    // cudaFree(convOutput_cpu);
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

    // cudaFree(inputTran_gpu);
    // cudaFree(filterTran_gpu);
    // cudaFree(gemmOutput_gpu);
    // printf("m1 before free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(*m1);
    // *m1 = nullptr;
    // printf("m1 after free valid:%d, addr:%d\n",(*m1!=NULL), *m1);
    // cudaFree(convOutput_gpu);
}


//OLD
// one thread corresponding to one layer
// and one block corresponding to one channel of all layer.

//NEW
// 1024 thread limit, hard to arrange the thread 
// one thread corresponding to one tile
// and one block corresponding to one tile of all batch.
// Problem: the thread acess is not continuely, and will cause serious delay problem, makes the problem super slow.
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs, int bound){
    int tidx = threadIdx.x;     // batch
    int bidx = blockIdx.x;     //  chn_in
    int bidy = blockIdx.y;     // blockn.x
    int bidz = blockIdx.z;     // blockn.y

    // if (tidx ==0 && tidy==0 && tidz==0 && bid ==0)
    //     printf("(bat4Conv,blockn,blockn,chn_in)=(%d,%d,%d,%d)\n",blockDim.z,blockDim.x,blockDim.y,gridDim.x);
    // printf("input_gpu[0]:%lf\n",pInputs[0]);

    int totalChn = gridDim.x;
    int numOfBatch = blockDim.x;
    int blockn = gridDim.y;

    float Mread[6][6] = {{0}};

    int inside = side;

    //take care of x-direction and y direction.
    // pInputs = &pInputs[bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)];

    // test delay.
    pInputs = &pInputs[bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidy*numOfBatch + bidz *numOfBatch *blockn + bidx *MSize];
    
    
    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j + i * inside];

                if (bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside) + j + i * inside >bound){
                    printf("exceed: %d,%d,%d,%d\n",bidx,bidy,bidz,tidx);
                }
            }
        }
    }

    // DEBUG
    // if (tidx ==0 && bidx==0 && bidz==0 && bidy ==5){
    //     for( int i =0;i<6;i++) {
    //         for (int j=0; j<6;j++) {
    //             // if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
    //                 // printf("%f  ",Mread [i][j]);
    //                 printf("(%d,%d,%d,address:%d)  ",(4*bidy+j),(4*bidz+i),(4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding),(bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)+j + i * inside));
    //         }printf("\n");
    //     }
    // }

    float Atd[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = MSize * KSize;

    #pragma unroll
    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}

// input: NCHW layout feature map.
// output: KM layout GEMM output, M is conposed of (batch, blockn, blockn)
__global__ void wino_input_trans_nchw_suitFor128_2(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs, int bound, int tileArray, int numOfBlcokn){
    int tidx = threadIdx.x; // blocknx
    int tidy = threadIdx.y; // blockny
    int bidx = blockIdx.x;  // bat4Conv
    int bidy = blockIdx.y;  // chn
    int bidz = blockIdx.z;  // blocknTile

    int numOfBatch = gridDim.x;
    int totalChn = gridDim.y;
    
    // the 2 here is a hyper-parameter, should be fixed in further developement.
    int blocknx = tidx + bidz% tileArray * 14;
    int blockny = tidy + bidz/ tileArray * 14;

    if( blocknx> numOfBlcokn || blockny > numOfBlcokn){
        if(bidx ==0 && bidy==0) {
            printf("true, %d %d %d %d %d\n",blocknx,blockny, bidz, tidx, tidy);
        }
        
        return;
    }
    float Mread[6][6] = {{0}};
    int inside = side;

    // __syncthreads();
    // if(tidx==0 && tidy==0 && bidx==0 && bidy ==0 && bidz==0){
    //     printf("src: %f %f %f %f %f %f %f\n",pInputs[0],pInputs[1],pInputs[2],pInputs[3],pInputs[4],pInputs[5],pInputs[6]);
    // }

    pInputs = &pInputs[bidx*totalChn*inside*inside + bidy*inside*inside + blocknx*4 + blockny*4*inside - padding*(1+inside)];
    // tbd, the last dim should be batch or blockn. It seems blockn could get better coalased access.
    pOutputs = &pOutputs[blocknx + blockny*numOfBlcokn + bidx*numOfBlcokn*numOfBlcokn + bidy*MSize];


    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            if( (4*blocknx +j >= padding)&& (4*blockny +i >= padding)&& (4*blocknx +j <= inside-1+padding)&& (4*blockny +i <= inside-1+padding)){
                Mread[i][j] = pInputs[j+i*inside];

                if(blocknx*4 + blockny*4*inside +bidy*inside*inside + bidx*inside*inside*totalChn-padding*(1+inside) + j +i*inside >bound){
                    printf("exceed: %d,%d,%d,%d,%d\n",tidx,tidy,bidx,bidy,bidz);
                }
            }
        }
    }
    // if(bidx ==0 && bidy==1 && blocknx==0 && blockny==0) {
    //     for (int i=0;i<6;i++){
    //         for(int j=0;j<6;j++){
    //             if( (4*blocknx +j >= padding)&& (4*blockny +i >= padding)&& (4*blocknx +j <= inside-1+padding)&& (4*blockny +i <= inside-1+padding)){
    //                 printf("%f ", pInputs[j+i*inside]);
    //             }else{
    //                 printf("skip  ");
    //             }
    //         }printf("\n");
    //     }
    // }

    

    // __syncthreads();
    // if(bidx ==0 && bidy==0 && blocknx==0 && blockny==0) {
    //     for (int i=0;i<6;i++){
    //         for(int j=0;j<6;j++){
    //             if( (4*blocknx +j >= padding)&& (4*blockny +i >= padding)&& (4*blocknx +j <= inside-1+padding)&& (4*blockny +i <= inside-1+padding)){
    //                 printf("%f ", Mread[i][j]);
    //             }else{
    //                 printf("skip  ");
    //             }
    //         }printf("\n");
    //     }
    // }

    // return;

    float Atd[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = MSize * KSize;

    #pragma unroll
    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}


__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // numOfFilter
    int bid  = blockIdx.x;   // chn_in

    int chn_in = gridDim.x;


    float Mread[3][3] ={{0}};
    pInputs = &pInputs[tidx*3*3*chn_in + bid *3*3];
    pOutputs = &pOutputs[tidx + bid*NSize];

    #pragma unroll
    for (int i= 0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Mread[i][j] = pInputs[j + i*3];
        }
    }

    float Gg[6][3] = {{0}};
    
    for(int i=0;i<3;i++){
        Gg[0][i] = Mread[0][i]/4;
        Gg[1][i] = -Mread[0][i]/6 - Mread[1][i]/6 - Mread[2][i]/6;
        Gg[2][i] = -Mread[0][i]/6 + Mread[1][i]/6 - Mread[2][i]/6;
        Gg[3][i] = Mread[0][i]/24 + Mread[1][i]/12 + Mread[2][i]/6;
        Gg[4][i] = Mread[0][i]/24 - Mread[1][i]/12 + Mread[2][i]/6;
        Gg[5][i] = Mread[2][i];
    }

    int size = NSize * KSize;

    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = Gg[i][0]/4;
        pOutputs[i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 5*size] = Gg[i][2];
    }
}


// input: MN layout GEMM output, M is conposed of (blockn, blockn, batch)
// output NCHW layout.
__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int chn = threadIdx.x;  // numOfFilter or chn_out
    int bidx = blockIdx.x; // bat4Conv 
    int bidy = blockIdx.y; // blockn.x
    int bidz = blockIdx.z; // blockn.y


    float Mread[6][6] = {{0}};
    //take care, this should be the total MSize and NSize, rather than M and N;
    int size = MSize * NSize;
    int chn_out = blockDim.x;
    int bat4Conv = gridDim.x;
    int blockn = gridDim.y;

    pInputs = &pInputs[chn + bidx* NSize + bidy* bat4Conv*NSize + bidz*bat4Conv*NSize*blockn];
    pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];

    for (int i=0;i<6;i++) {
        for (int j=0; j<6; j++) {
            Mread[i][j] = pInputs[j*size + i*6*size];
        }
    }

    //output
    float Atd[4][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
        Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
        Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
    }

    // float cache[4][4] = {{0}};
    for(int i=0;i<4;i++){
        Mread[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        Mread[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        Mread[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        Mread[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        // pOutputs[i*oside*bat4Conv] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        // pOutputs[i*oside*bat4Conv + bat4Conv] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 2*bat4Conv] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 3*bat4Conv] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*bidy+j<=oside-1 ) && ( 4*bidz+i<=oside-1 ) ){
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }
}


// input: MN layout GEMM output, M is conposed of (batch, blockn, blockn)
// output NCHW layout.
__global__ void wino_invers_nchw_suitFor128_2(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int chn = threadIdx.x;  // numOfFilter or chn_out
    int bidx = blockIdx.x; // bat4Conv 
    int bidy = blockIdx.y; // blockn.x
    int bidz = blockIdx.z; // blockn.y


    float Mread[6][6] = {{0}};
    //take care, this should be the total MSize and NSize, rather than M and N;
    int size = MSize * NSize;
    int chn_out = blockDim.x;
    int bat4Conv = gridDim.x;
    int blockn = gridDim.y;

    // pInputs = &pInputs[chn + bidx* NSize + bidy* bat4Conv*NSize + bidz*bat4Conv*NSize*blockn];
    pInputs = &pInputs[chn + bidy* NSize + bidz* blockn*NSize + bidx*NSize*blockn*blockn ];
    pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];

    for (int i=0;i<6;i++) {
        for (int j=0; j<6; j++) {
            Mread[i][j] = pInputs[j*size + i*6*size];
        }
    }

    __syncthreads();
    // if(chn ==0 && bidx==0 &&bidy==0 && bidz==0){
    //     printf("Mread:\n");
    //     for(int i=0;i<6;i++) {
    //         for (int j=0;j<6;j++){
    //             printf("%f ", Mread[i][j]);
    //         }printf("\n");
    //     }
    // }
    
    //output
    float Atd[4][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
        Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
        Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
    }

    // float cache[4][4] = {{0}};
    for(int i=0;i<4;i++){
        Mread[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        Mread[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        Mread[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        Mread[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        // pOutputs[i*oside*bat4Conv] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        // pOutputs[i*oside*bat4Conv + bat4Conv] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 2*bat4Conv] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 3*bat4Conv] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*bidy+j<=oside-1 ) && ( 4*bidz+i<=oside-1 ) ){
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }

    // if(chn==0 && bidx==0 && bidy==0 && bidz==0){
    //     printf("dst: %f %f %f %f\n",Mread[0][0],Mread[0][1],Mread[0][2],Mread[0][3]);
    // }
}


// input: NM layout GEMM output, M is conposed of (batch, blockn, blockn)
// output NCHW layout.
__global__ void wino_invers_nchw_suitFor128_3(int oside, int MSize, int NSize, float* pInputs, float* pOutputs, int tileArray, int numOfBlcokn){
    int tidx = threadIdx.x; // blockn.x
    int tidy = threadIdx.y; // blockn.y
    int bidx = blockIdx.x;  // bat4Conv
    int bidy = blockIdx.y;  // numOfFilter or chn_out
    int bidz = blockIdx.z;  // blocknTile
    int chn_out = gridDim.y;

    // the 2 here is a hyper-parameter, should be fixed in further developement.
    int blocknx = tidx + bidz% tileArray * 14;
    int blockny = tidy + bidz/ tileArray * 14;

    if( blocknx> numOfBlcokn || blockny > numOfBlcokn){
        return;
    }    

    float Mread[6][6] = {{0}};
    //take care, this should be the total MSize and NSize, rather than M and N;
    int size = MSize * NSize;
    
    // int bat4Conv = gridDim.x;


    // pInputs = &pInputs[chn + bidy* NSize + bidz* blockn*NSize + bidx*NSize*blockn*blockn ];
    // pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];
    pInputs = &pInputs[blocknx + blockny*numOfBlcokn + bidx*numOfBlcokn*numOfBlcokn + bidy*MSize ];
    pOutputs = &pOutputs[blocknx*4 + blockny*4*oside + bidx*oside*oside*chn_out + bidy*oside*oside];

    for (int i=0;i<6;i++) {
        for (int j=0; j<6; j++) {
            Mread[i][j] = pInputs[j*size + i*6*size];
        }
    }

    // __syncthreads();
    // if(tidx ==0 && tidy==0 && bidx==0 &&bidy==14 && bidz==0){
    //     printf("MSize:%d\n",MSize);
    //     printf("blocknx:%d blockny:%d bidx:%d bidy:%d bidz:%d\n",blocknx, blockny, bidx, bidy, bidz);
    //     // printf("Mread:\n");
    //     // for(int i=0;i<6;i++) {
    //     //     for (int j=0;j<6;j++){
    //     //         printf("%f ", Mread[i][j]);
    //     //     }printf("\n");
    //     // }
    // }

    // __syncthreads();
    //output
    float Atd[4][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
        Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
        Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
    }

    // float cache[4][4] = {{0}};
    for(int i=0;i<4;i++){
        Mread[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        Mread[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        Mread[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        Mread[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        // pOutputs[i*oside*bat4Conv] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        // pOutputs[i*oside*bat4Conv + bat4Conv] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 2*bat4Conv] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 3*bat4Conv] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*blocknx+j<=oside-1 ) && ( 4*blockny+i<=oside-1 ) ){
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }

    // if(tidx ==4 && tidy==2 && bidx==0 &&bidy==14 && bidz==0){
    //     printf("dst: %f %f %f %f\n",Mread[0][0],Mread[0][1],Mread[0][2],Mread[0][3]);
    // }
}