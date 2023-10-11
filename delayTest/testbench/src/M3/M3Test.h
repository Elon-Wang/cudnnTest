#include "testCase.h"
#include "GEMM.cuh"

__global__ void warmup(){}

class GEMM_Method{

    public:
    int min_M;
    int max_M;
    int min_N;
    int max_N;
    int min_K;
    int max_K;
    int maxBatch;

    // int blockn;
    // int inside_beta;

    int bat4Gemm =36;
    int M, N, K;
    int MCheck, NCheck, KCheck;
    int MSize, NSize, KSize;
    int blockx, blocky;


    // int padding =1;
    // int marginOfInputSide, M, K;
    // bool sideCheck, MCheck, KCheck;
    int nGemmOutput;
    float* gemmOutput_gpu;
    float* gemmOutput_cpu;
    char outFileName[30] = "default_Name.bin";

    // counted in micro second
    float singleTime =0;
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid= false;

    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    virtual void execut(testCase tc) =0 ;
    inputTransMethod(){
        // minSide = 4;
        // maxSide = 1024;
        // maxChannel = 512;
        // maxBatch = 1024;
        min_M = 128;
        max_M = 128;
        min_K = 4;
        min_N = 128;
        max_K = 65000;
        max_N = 65000;
        // printf("baseClasse default constructor\n minSide:%d, maxSide:%d maxChannel:%d maxBatch:%d\n",minSide, maxSide, maxChannel, maxBatch);
    }
};

bool GEMM_Method::testValid(testCase tc){
    M = tc.M;
    N = tc.N;
    K = tc.K;
    MCheck = ( M % 128 == 0 )? true: false;
    NCheck = ( N % 128 == 0 )? true: false;
    KCheck = ( chn % 8 == 0 )? true: false;
    MSize = MCheck? M: ((M/128 +1) * 128);
    NSize = NCheck? N: ((N/128 +1) * 128);
    KSize = KCheck? chn: ((chn/8 +1) * 8);

    
    if(tc.M >= min_M && tc.M <=max_M && tc.N >= min_N&& tc.N <= max_N && tc.K >= min_K && tc.K <= max_K) {
        blockx = (M+127)/128;
        blocky = (N+127)/128;
        nGemmOutput = 36*M*N;
        cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);
        gemmOutput_cpu = (float *)malloc(nGemmOutput *sizeof(float));

        return true;
    } else{
        printf("**************test case invalid*************\n");
        return false;
    }
}

float GEMM_Method::testPerformance(testCase tc){
    cudaEvent_t start1,stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, NULL);
    execut(tc);
    cudaEventRecord(stop1, NULL);

    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);

    cudaEventElapsedTime(&singleTime, start1, stop1);
    
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    cudaMemcpy( gemmOutput_cpu, gemmOutput_gpu, nGemmOutput, cudaMemcpyDeviceToHost);
    printf("first element:%f\n",gemmOutput_cpu[0]);
    save_parameter(outFileName, nGemmOutput, gemmOutput_cpu);

    return singleTime;
}

class new0 : public GEMM_Method{
    public:
    virtual void execut(testCase tc){
        GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, tc.inputTran_gpu, tc.kernelTran_gpu,0, gemmOutput_gpu);
    }
}
