// #pragma once
#include "testCase.h"
// #include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);


class inputTransMethod{

    public:
    int minSide;
    int maxSide;
    int maxChannel;
    int maxBatch;
    int MSize;
    int KSize;
    int blockn;
    int inside_beta;
    int nInputTran;
    int padding =1;
    float singleTime =0;
    float* inputTran_gpu;
    float* inputTran_cpu;

    // counted in micro second
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid= false;

    bool testValid(testCase tc);
    void testPerformance(testCase tc);
    virtual void execut(testCase tc) =0 ;
    inputTransMethod(){
        minSide = 4;
        maxSide = 1024;
        maxChannel = 512;
        maxBatch = 1024;
        // printf("baseClasse default constructor\n minSide:%d, maxSide:%d maxChannel:%d maxBatch:%d\n",minSide, maxSide, maxChannel, maxBatch);
    }
};

class new1 : public inputTransMethod{
    public:
    virtual void execut(testCase tc) {
        // printf("chn:%d\n",tc.chn);
        // printf("KSize:%d MSize:%d blockn:%d bat4conv:%d inside:%d, \n",KSize, MSize, blockn, tc.bat4Conv, tc.inside);
        wino_input_trans_nchw_suitFor128<<<dim3(KSize,blockn,blockn),dim3(tc.bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};



bool inputTransMethod::testValid(testCase tc){
    // printf("tc.inside:%d tc.chn:%d tc.bat4Conv:%d  maxSide:%d maxBatch:%d\n",tc.inside, tc.chn, tc.bat4Conv, maxSide, maxBatch);
    // printf("%d, %d, %d, %d\n", (tc.inside >=minSide), (tc.inside <=maxSide), (tc.chn<=maxChannel), (tc.bat4Conv<=maxBatch));
    if(tc.inside >=minSide && tc.inside <=maxSide && tc.chn<=maxChannel && tc.bat4Conv<=maxBatch) {
        // inside_beta = (tc.side-2)%4: 4+(tc.side-2)%4+tc.side? tc.side;
        inside_beta = tc.inside;
        blockn = (inside_beta-2)/4;
        MSize = blockn*blockn*tc.bat4Conv;
        KSize = tc.chn;
        
        // printf("KSize:%d MSize:%d blockn:%d bat4conv:%d inside:%d, \n",KSize, MSize, blockn, tc.bat4Conv, tc.inside);
        nInputTran = 36 * MSize*KSize;

        cudaMalloc((void**) &inputTran_gpu, nInputTran<<2);
        inputTran_cpu = (float *) malloc(nInputTran * sizeof(float) );
        return true;
    }   else {
        printf("Valid False\n");
        return false;
    }
}

void inputTransMethod::testPerformance(testCase tc){
    
    
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

    printf("the time of this execut is %f\n",singleTime);
    
    cudaMemcpy(inputTran_cpu, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    printf("first element:%f\n",inputTran_cpu[0]);
    // profile the timing of the program
    
    // avetime = avetime/10;
}



class new2 : public inputTransMethod{
public:
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran1_gpu);
    }
};

class new3 : public inputTransMethod{
public:
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran1_gpu);
    }
};

class new4 : public inputTransMethod{
public:
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran1_gpu);
    }
};



__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
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
    pInputs = &pInputs[bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidy*numOfBatch + bidz *numOfBatch *blockn + bidx *MSize];
    
    
    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j + i * inside];
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