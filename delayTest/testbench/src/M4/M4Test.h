#include "testCase.h"

class inverse_Method{

    int inside, oside;
    int M,N,K;
    int MCheck, NCheck;
    int MSize, NSize;
    int bat4Conv;
    int numOfFilter;
    int blockn;
    int nOutput;
    float *output_cpu;
    float *output_gpu;

    float minDelay;
    float singleTime;

    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    virtual void execut(testCase tc)=0;
}

bool inverse_Method::testValid(testCase tc){
    bat4Conv = tc.bat4Conv;
    numOfFilter = tc.numOfFilter;
    inside = tc.inside;
    oside = inside +2*padding -2;
    N = numOfFilter;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);


    if(){

        blockn = (inside_beta -2) /4;
        MCheck = ( M % 128 == 0 )? true: false;
        NCheck = ( N % 128 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        NSize = NCheck? N: ((N/128 +1) * 128);

        nOutput = ;//batch *oside*oside *numOfFilter;
        cudaMalloc((void **) &output_gpu, nOutput<<2);
        output_cpu = (float *) malloc(nOutput*sizeof(float));
        return true;
    } else {
        return false;
    }

}

float inverse_Method::testPerformance(testCase tc){
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

    // printf("the time of this execut is %f\n",singleTime);
    
    cudaMemcpy(output_cpu, output_gpu, nOutput<<2, cudaMemcpyDeviceToHost);
    printf("first element:%f\n",output_cpu[0]);
    save_parameter(outFileName, nOutput, output_cpu);

    return singleTime;
}

class new0 : public inverse_Method{
    public:
    virtual void execut(testCase tc){
        wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, tc.gemmOutput_gpu, output_gpu);
    }
}