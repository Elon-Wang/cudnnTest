#include "testCase.h"

__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);

class inverse_Method{
    public:
    int inside_beta, oside;
    int M,N,K;
    int MCheck, NCheck;
    int MSize, NSize;
    int bat4Conv;
    int numOfFilter;
    int blockn;
    int nOutput;
    float *output_cpu;
    float *output_gpu;


    char outFileName[30] = "default_Name.bin";
    float minDelay;
    float singleTime;
    bool valid;

    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    virtual void execut(testCase tc)=0;

    inverse_Method(){
        //initial
    }
};

bool inverse_Method::testValid(testCase tc){
    bat4Conv = tc.bat4Conv;
    numOfFilter = tc.numOfFilter;
    oside = tc.oside;
    M = tc.M;
    N = numOfFilter;
    inside_beta = tc.inside_beta;

    if(true){

        blockn = (inside_beta -2) /4;
        MCheck = ( M % 128 == 0 )? true: false;
        NCheck = ( N % 128 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        NSize = NCheck? N: ((N/128 +1) * 128);

        nOutput = bat4Conv * oside * oside * numOfFilter;//batch *oside*oside *numOfFilter;
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
    new0(){
        strcpy(outFileName, "./data/M4_new0.bin");
    }
    virtual void execut(testCase tc){
        // printf("bat4Conv:%d, blockn:%d, numOfFilter:%d\n", bat4Conv, blockn, numOfFilter);
        // wino_invers_nchw_suitFor128<<<dim3(1,1,1), dim3(1,1,1)>>>(oside, MSize, NSize, tc.gemmOutput_gpu, output_gpu);
        wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, tc.gemmOutput_gpu, output_gpu);
    }
};

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

    // if (chn ==0 && bidx ==0 && bidy ==0 && bidz==0) {
    //     printf("oside:%d\n",oside);
    //     for (int i=0;i<4;i++) {
    //         for (int j=0; j<4; j++) {
    //             printf("%d ",i*oside + j);
    //         }
    //         printf("\n");
    //     }
    // }
    

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*bidy+j<=oside-1 ) && ( 4*bidz+i<=oside-1 ) ){
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }
}