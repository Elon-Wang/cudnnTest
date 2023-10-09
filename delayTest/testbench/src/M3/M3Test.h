#include "testCase.h"

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
    int blockn;
    int inside_beta;
    int nGemmOutput;
    
    float singleTime =0;

    // int padding =1;
    // int marginOfInputSide, M, K;
    // bool sideCheck, MCheck, KCheck;

    float* gemmOutput_gpu;
    float* gemmOutput_cpu;
    char outFileName[30] = "default_Name.bin";

    // counted in micro second
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
    if(tc.M >= min_M && tc.M <=max_M && tc.N >= min_N&& tc.N <= max_N && tc.K >= min_K && tc.K <= max_K) {

        nGemmOutput = 36*
        return true;
    } else{
        printf("**************test case invalid*************\n");
        return false;
    }
}

