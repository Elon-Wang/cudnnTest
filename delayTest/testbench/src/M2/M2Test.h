#include "testCase.h"

__global__ void warmup(){}
__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);

class kernelTranMethod{
public:
    int min_numOfFilter;
    int max_numOfFilter;
    int min_chn;
    int max_chn;
    int nKernelTran;
    int N, K;
    int NSize, KSize;
    bool NCheck, KCheck;

    float *kernelTran_cpu;
    float *kernelTran_gpu;
    char outFileName[40] = "default_Name.bin";
    char fileLastName[20] = "default_LastName";

    float singleTime;
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid = false;
    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    void reportPerformance(testCase tc);
    virtual void execut(testCase tc)=0 ;
};

bool kernelTranMethod::testValid(testCase tc){
    if(tc.numOfFilter >= min_numOfFilter && tc.numOfFilter <= max_numOfFilter && tc.chn ){
        N = tc.numOfFilter;
        K = tc.chn;
        NCheck = ( N % 128 == 0 )? true: false;
        KCheck = ( K % 8 == 0 )? true: false;
        NSize = NCheck? N: ((N/128 +1) * 128);
        KSize = KCheck? K: ((K/8 +1) * 8);

        nKernelTran = 36*NSize *KSize;
        cudaMalloc((void **) &kernelTran_gpu, nKernelTran<<2);        
        kernelTran_cpu = (float*)malloc(nKernelTran *sizeof(float));

        return true;
    } else{
        printf("**************test case invalid*************\n");
        return false;
    }
}

float kernelTranMethod::testPerformance(testCase tc) {

    warmup<<<1,1>>>();
    int cnt =10;
    float timeSeries[cnt];
    avgDelay = 0;
    char tcidx[5];
    sprintf(tcidx,"%d",tc.index);
    strcat(outFileName, tcidx);
    strcat(outFileName, fileLastName);

    for(int i =0; i < cnt;i++) {
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

        timeSeries[i] = singleTime;
        avgDelay += singleTime;
    }
    
    avgDelay /= cnt;
    minDelay = timeSeries[0];
    maxDelay = timeSeries[0];
    for (int i=1; i<cnt; i++){
        maxDelay = (maxDelay > timeSeries[i])? maxDelay: timeSeries[i];
        minDelay = (minDelay < timeSeries[i])? minDelay: timeSeries[i];
    }

    cudaMemcpy(kernelTran_cpu, kernelTran_gpu, nKernelTran<<2, cudaMemcpyDeviceToHost);
    // printf("first element:%f\n",kernelTran_cpu[0]);
    save_parameter(outFileName, nKernelTran, kernelTran_cpu);
    return avgDelay;
}

void kernelTranMethod::reportPerformance(testCase tc){
    printf("testCase:%d\t avgDelay:%f\tminDelay:%f\tmaxDelay:%f\n",tc.index, avgDelay, minDelay,maxDelay);
}

// original design is assigned as new0;
class new0 : public kernelTranMethod{
public:
    new0(){
        min_numOfFilter = 1;
        max_numOfFilter = 65525;
        min_chn = 1;
        max_chn = 65525;
        strcpy(fileLastName,"/M2_new0.bin");
        strcpy(outFileName, "./data/tc");
    }
    virtual void execut(testCase tc) {
        // printf("%d %d\n",tc.chn, tc.numOfFilter);
        wino_kernel_trans_nchw_suitFor128<<<dim3(tc.chn,1,1),dim3(tc.numOfFilter,1,1)>>>(NSize, KSize, tc.kernel_gpu, kernelTran_gpu);
    }
};


__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // numOfFilter
    int bid  = blockIdx.x;   // chn_in

    // int chn_out = blockDim.x;


    float Mread[3][3] ={{0}};
    pInputs = &pInputs[tidx*3*3*gridDim.x + bid *3*3];
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

    // if (bid==0 && tidx ==  16) {
    //     printf("input pos:%d\n",tidx*3*3*gridDim.x + bid *3*3);
    //     printf("val:%f\n",pInputs[tidx*3*3*gridDim.x + bid *3*3]);
    // }

    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = Gg[i][0]/4;
        pOutputs[i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 5*size] = Gg[i][2];
    }
}