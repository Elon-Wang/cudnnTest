// #pragma once
#include "testCase.h"
#include "string"
// #include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nhwc_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new1(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new2(int side, int side_beta, int MSize, int KSize, int padding, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new4(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new5(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nhwc_suitFor128_new6(int side, int side_beta, int totalChn, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);

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
    

    int marginOfInputSide, M, K;
    bool sideCheck, MCheck, KCheck;

    float* inputTran_gpu;
    float* inputTran_cpu;
    char outFileName[40] = "default_Name.bin";
    char fileLastName[20] = "default_LastName";

    // counted in micro second
    float singleTime =0;
    float minDelay;
    float maxDelay;
    float avgDelay;
    float totalDelay;
    bool valid= false;

    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    virtual void execut(testCase tc) =0 ;
    void reportPerformance(testCase tc);
    inputTransMethod(){
        minSide = 4;
        maxSide = 1024;
        maxChannel = 512;
        maxBatch = 1024;
        // printf("baseClasse default constructor\n minSide:%d, maxSide:%d maxChannel:%d maxBatch:%d\n",minSide, maxSide, maxChannel, maxBatch);
    }
    // ~inputTransMethod(){
    //     free(inputTran_gpu);
    //     free(inputTran_cpu);
    // }
};

bool inputTransMethod::testValid(testCase tc){
    // printf("tc.inside:%d tc.chn:%d tc.bat4Conv:%d  maxSide:%d maxBatch:%d\n",tc.inside, tc.chn, tc.bat4Conv, maxSide, maxBatch);
    // printf("%d, %d, %d, %d\n", (tc.inside >=minSide), (tc.inside <=maxSide), (tc.chn<=maxChannel), (tc.bat4Conv<=maxBatch));
    if(tc.inside >=minSide && tc.inside <=maxSide && tc.chn<=maxChannel && tc.bat4Conv<=maxBatch) {
        // inside_beta = (tc.side-2)%4: 4+(tc.side-2)%4+tc.side? tc.side;

        marginOfInputSide = (tc.inside+2*padding-6)%4;
        sideCheck = ( marginOfInputSide == 0 )? true: false ;
        inside_beta = sideCheck? tc.inside +2*padding : (tc.inside + 2*padding + 4 - marginOfInputSide);

        blockn = (inside_beta -2) /4;

        // inside_beta = tc.inside;
        // blockn = (inside_beta-2)/4;
        // MSize = blockn*blockn*tc.bat4Conv;
        // KSize = tc.chn;

        M = tc.bat4Conv * blockn * blockn;
        K = tc.chn;

        MCheck = ( M % 128 == 0 )? true: false;
        KCheck = ( K % 8 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        KSize = KCheck? K: ((K/8 +1) * 8);
        
        // printf("KSize:%d MSize:%d blockn:%d bat4conv:%d inside:%d, \n",KSize, MSize, blockn, tc.bat4Conv, tc.inside);
        nInputTran = 36 * MSize*KSize;

        cudaMalloc((void**) &inputTran_gpu, nInputTran<<2);
        inputTran_cpu = (float *) malloc(nInputTran * sizeof(float) );
        return true;
    } else {
        printf("Valid False\n");
        return false;
    }
}

float inputTransMethod::testPerformance(testCase tc){
    
    warmup<<<1,1>>>();
    int cnt =2;
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
    
    totalDelay = avgDelay;
    avgDelay /= cnt;
    minDelay = timeSeries[0];
    maxDelay = timeSeries[0];
    for (int i=1; i<cnt; i++){
        maxDelay = (maxDelay > timeSeries[i])? maxDelay: timeSeries[i];
        minDelay = (minDelay < timeSeries[i])? minDelay: timeSeries[i];
    }
    // printf("the time of this execut is %f\n",singleTime);
    
    cudaMemcpy(inputTran_cpu, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // printf("first element:%f\n",inputTran_cpu[0]);
    // printf("%s\n",outFileName);
    save_parameter(outFileName, nInputTran, inputTran_cpu);

    return avgDelay;
    // profile the timing of the program
}

void inputTransMethod::reportPerformance(testCase tc){
    printf("testCase:%d\t total:%f \tavgDelay:%f\tminDelay:%f\tmaxDelay:%f\n",tc.index, totalDelay,avgDelay, minDelay,maxDelay);
}

class orig : public inputTransMethod{
    public:
    orig(){
        maxBatch =1024;
        minSide = 6;
        maxSide = 262142;
        maxChannel = 512000;
        strcpy(fileLastName,"/M1_new0.bin");
        strcpy(outFileName, "./data/tc");
    }
    virtual void execut(testCase tc) {
        wino_input_trans_nhwc_suitFor128<<<dim3(blockn, blockn, tc.bat4Conv),dim3(tc.chn,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new1 : public inputTransMethod{
    public:
    new1(){
        minSide = 6;
        maxSide = 69632;
        maxBatch = 65535;
        maxChannel = 512000;
        strcpy(fileLastName,"/M1_new1.bin");
        strcpy(outFileName, "./data/tc");
    }
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        // printf("blockn:%d, blockn_beta:%d\n",blockn_makeup,blockn);
        wino_input_trans_nhwc_suitFor128_new1<<<dim3(blockn, blockn, tc.bat4Conv),dim3(tc.chn,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new2 : public inputTransMethod{
    public:
    new2(){
        maxChannel = 1024;
        minSide = 6;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(fileLastName,"/M1_new2.bin");
        strcpy(outFileName, "./data/tc");
    } 
    virtual void execut(testCase tc) {
        wino_input_trans_nhwc_suitFor128_new2<<<dim3(blockn,blockn,tc.bat4Conv),dim3(tc.chn)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new3 : public inputTransMethod{
    public:
    new3(){
        maxChannel = 65535;
        minSide = 6;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(fileLastName,"/M1_new3.bin");
        strcpy(outFileName, "./data/tc");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        int chn_makeup = (tc.chn%4)? (tc.chn+4-tc.chn%4):tc.chn;
        wino_input_trans_nhwc_suitFor128_new3<<<dim3(blockn , blockn, tc.bat4Conv),dim3(tc.chn,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new4 : public inputTransMethod{
    public:
    new4(){
        maxChannel = 65535;
        minSide = 6;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(fileLastName,"/M1_new4.bin");
        strcpy(outFileName, "./data/tc");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        M = tc.bat4Conv * blockn_makeup * blockn_makeup;
        MSize = (M % 128 == 0)? M: ((M/128 +1) * 128);
        wino_input_trans_nhwc_suitFor128_new4<<<dim3(blockn*blockn,(tc.chn/64), tc.bat4Conv),dim3(64, 6)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new5 : public inputTransMethod{
    public:
    new5(){
        maxChannel = 65535;
        minSide = 6;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(fileLastName,"/M1_new5.bin");
        strcpy(outFileName, "./data/tc");
    } 
    virtual void execut(testCase tc) {
        wino_input_trans_nhwc_suitFor128_new5<<<dim3(blockn, blockn, tc.bat4Conv),dim3(tc.chn,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new6 : public inputTransMethod{
    public:
    new6(){
        maxChannel = 65535;
        minSide = 6;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(fileLastName,"/M1_new6.bin");
        strcpy(outFileName, "./data/tc");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        // printf("blockn:%d\tM:%d\tMSize:%d\n",blockn,M,MSize);
        // M = tc.bat4Conv * blockn_makeup * blockn_makeup;
        // MSize = (M % 128 == 0)? M: ((M/128 +1) * 128);
        // printf("blockn:%d\tM:%d\tMSize:%d\n",blockn_makeup,M,MSize);
        if (tc.chn<63){
            wino_input_trans_nhwc_suitFor128_new5<<<dim3(blockn, blockn, tc.bat4Conv),dim3(tc.chn,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
            // wino_input_trans_nhwc_suitFor128_new6<<<dim3(blockn*blockn,1, tc.bat4Conv),dim3(tc.chn, 6)>>>(tc.inside, inside_beta, tc.chn, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
        } else {
            wino_input_trans_nhwc_suitFor128_new6<<<dim3(blockn*blockn,((tc.chn+63)/64), tc.bat4Conv),dim3(64, 6)>>>(tc.inside, inside_beta, tc.chn, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
        }
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

__global__ void wino_input_trans_nhwc_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // chn_in
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int totalChn = blockDim.x;
    int blockn = gridDim.x;
    int numOfBatch = gridDim.z;

    float Mread[6][6] = {{0}};

    int inside = side;

    // take care of x-direction and y direction.
    pInputs = &pInputs[tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn - padding*(padding+inside)*totalChn ];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidz*KSize + bidx*numOfBatch*KSize + bidy*numOfBatch*blockn*KSize];
    // pOutputs = &pOutputs[tidx + bidx*numOfBatch + bidy*numOfBatch*blockn + bidx*MSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
            }
        }
    }

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

__global__ void wino_input_trans_nhwc_suitFor128_new1(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // chn_in
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int totalChn = blockDim.x;
    int blockn = gridDim.x;
    int numOfBatch = gridDim.z;

    float Mread[6][6] = {{0}};

    int inside = side;

    // take care of x-direction and y direction.
    pInputs = &pInputs[tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn - padding*(padding+inside)*totalChn ];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidz*KSize + bidx*numOfBatch*KSize + bidy*numOfBatch*blockn*KSize];
    // pOutputs = &pOutputs[tidx + bidx*numOfBatch + bidy*numOfBatch*blockn + bidx*MSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside

    if (bidx==0 || bidy==0 || bidx == blockn-1 || bidy == blockn-1 ){
        for( int i =0;i<6;i++) {
            for (int j=0; j<6;j++) {
                if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                    Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
                }
            }
        }
    } else {
        for( int i =0;i<6;i++) {
            for (int j=0; j<6;j++) {
                Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
            }
        }
    }

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

__global__ void wino_input_trans_nhwc_suitFor128_new2(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // chn_in
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int totalChn = blockDim.x;
    int blockn = gridDim.x;
    int numOfBatch = gridDim.z;

    float Mread[6][6] = {{0}};

    int inside = side;

    // take care of x-direction and y direction.
    pInputs = &pInputs[tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidz*KSize + bidx*numOfBatch*KSize + bidy*numOfBatch*blockn*KSize];
    // pOutputs = &pOutputs[tidx + bidx*numOfBatch + bidy*numOfBatch*blockn + bidx*MSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            // if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
            // }
        }
    }

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

__global__ void wino_input_trans_nhwc_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx  = threadIdx.x;  // total chn is is 512, chn
    int bidx = blockIdx.x;    // total row size is 4, row
    int bidy = blockIdx.y;    // total col size is 4, col
    int bidz = blockIdx.z;    //total is 1, batch

    int Inside = side;
    int totalChn = blockDim.x;
    int numOfBatch =  gridDim.z;
    int blockn = 1 + (Inside-6)/4;

    float Mread[6][6];

    int input_loc = tidx + bidx*4*totalChn + bidy*Inside*4*totalChn + bidz*Inside*Inside*totalChn;

    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            Mread[i][j] = pInputs[input_loc + j*totalChn + i*Inside*totalChn];           
        }
    }
    
    // 16 is the location, the tilesize
    int output_loc = bidz*blockn*blockn*totalChn + tidx + bidx*totalChn + bidy*blockn*totalChn;

    float Atd[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = numOfBatch*blockn*blockn*totalChn;

    for(int i=0;i<6;i++){
        pOutputs[output_loc + i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[output_loc + i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}


__global__ void wino_input_trans_nhwc_suitFor128_new4(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int Inside = side;
    int blockn = 1 + (Inside-6)/4;
    
    int tidx = threadIdx.x; // each block deal with 64 chn
    int tidy = threadIdx.y;  // use to parallem the trans 6

    int blocknx = blockIdx.x%blockn; // total row size, row
    int blockny = blockIdx.x/blockn; // total col size, col


    int bidy = blockIdx.y; //spilit chn (chn/64)
    int bidz = blockIdx.z;// number of graph
    
    
    int totalChn = blockDim.x*gridDim.y;
    int numOfBatch = gridDim.z;
    

    __shared__ float Mread[6][6][64];
    __shared__ float Atd[6][6][64];

    int input_loc = tidx + bidy*64 + blocknx*4*totalChn + blockny*Inside*4*totalChn + bidz*Inside*Inside*totalChn + tidy*totalChn;

    for(int i=0;i<6;i++){
        Mread[i][tidy][tidx] = pInputs[input_loc + i*Inside*totalChn];           
    }
    __syncthreads();

    int output_loc = bidz*blockn*blockn*totalChn + tidx + bidy*64 + blocknx*totalChn + blockny*blockn*totalChn;

    Atd[0][tidy][tidx] =  4*Mread[0][tidy][tidx] - 5*Mread[2][tidy][tidx] +   Mread[4][tidy][tidx];
    Atd[1][tidy][tidx] = -4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] +   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[2][tidy][tidx] =  4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] -   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[3][tidy][tidx] = -2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] + 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[4][tidy][tidx] =  2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] - 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[5][tidy][tidx] =  4*Mread[1][tidy][tidx] - 5*Mread[3][tidy][tidx] +   Mread[5][tidy][tidx];

    __syncthreads();
    
    int size = numOfBatch * blockn * blockn * totalChn;

    pOutputs[output_loc + tidy*6*size]          =  4*Atd[tidy][0][tidx] - 5*Atd[tidy][2][tidx] +   Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + size]   = -4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] +   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 2*size] =  4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] -   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 3*size] = -2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] + 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 4*size] =  2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] - 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 5*size] =  4*Atd[tidy][1][tidx] - 5*Atd[tidy][3][tidx] +   Atd[tidy][5][tidx];
    __syncthreads();
}

__global__ void wino_input_trans_nhwc_suitFor128_new5(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // chn_in
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int inside = side;
    int totalChn = blockDim.x;
    int blockn = gridDim.x;

    // int marginOfInputSide = (inside+2*1-6)%4;
    // bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    // int inside_beta = sideCheck? inside +2*1 : ( inside + 2*1 + 4 - marginOfInputSide);
    
    // int blockn = (inside -2) /4;
    int numOfBatch = gridDim.z;

    float Mread[6][6];

    // take care of x-direction and y direction.
    // int input_loc = tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn;
    // int output_loc = tidx + bidx*KSize + bidy*blockn*KSize + bidz*blockn*blockn*KSize;
    pInputs = &pInputs[tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn- padding*(padding+inside)*totalChn];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    // pOutputs = &pOutputs[tidx + bidz*KSize + bidx*numOfBatch*KSize + bidy*numOfBatch*blockn*KSize];
    pOutputs = &pOutputs[tidx + bidx*KSize+ bidy*blockn*KSize+ bidz*blockn*blockn*KSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside
    // if (bidx==0 || bidy==0 || bidx == blockn-1 || bidy == blockn-1 ){
        for( int i =0;i<6;i++) {
            for (int j=0; j<6;j++) {
                if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                    Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
                }
            }
        }
    // } else {
    //     for( int i =0;i<6;i++) {
    //         for (int j=0; j<6;j++) {
    //             Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
    //         }
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
    // int size = numOfBatch*blockn*blockn*totalChn;

    #pragma unroll
    for(int i=0;i<6;i++){
        pOutputs[ i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[ i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[ i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[ i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[ i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[ i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}

__global__ void wino_input_trans_nhwc_suitFor128_new6(int side, int side_beta, int totalChn, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int Inside = side;
    int blockn = (side_beta -2) /4;
    
    int tidx = threadIdx.x; // chn, 64 at maximum
    int tidy = threadIdx.y; // use to parallem the trans 6

    int blocknx = blockIdx.x%blockn; // blockn.x
    int blockny = blockIdx.x/blockn; // blockn.y


    int bidy = blockIdx.y;  // spilit chn (chn/64)
    int bidz = blockIdx.z;  // batch
    
    
    // int totalChn = blockDim.x*gridDim.y;
    int numOfBatch = gridDim.z;
    
    if (bidy*64+tidx >=totalChn){
        return;
    }
    __shared__ float Mread[6][6][64];
    __shared__ float Atd[6][6][64];

    int input_loc = tidx + bidy*64 + blocknx*4*totalChn + blockny*Inside*4*totalChn + bidz*Inside*Inside*totalChn + tidy*totalChn - padding*(padding+Inside)*totalChn;

    for(int i=0;i<6;i++){
        if((4*blocknx+tidy >= padding) && (4*blockny+i >= padding) && (4*blocknx+tidy <= Inside-1+padding) && (4*blockny+i <= Inside-1+padding)) {
            Mread[i][tidy][tidx] = pInputs[input_loc + i*Inside*totalChn];
        }
    }
    __syncthreads();

    int output_loc = bidz*blockn*blockn*KSize + tidx + bidy*64 + blocknx*KSize + blockny*blockn*KSize;

    Atd[0][tidy][tidx] =  4*Mread[0][tidy][tidx] - 5*Mread[2][tidy][tidx] +   Mread[4][tidy][tidx];
    Atd[1][tidy][tidx] = -4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] +   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[2][tidy][tidx] =  4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] -   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[3][tidy][tidx] = -2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] + 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[4][tidy][tidx] =  2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] - 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[5][tidy][tidx] =  4*Mread[1][tidy][tidx] - 5*Mread[3][tidy][tidx] +   Mread[5][tidy][tidx];

    __syncthreads();
    
    int size = MSize * KSize;

    pOutputs[output_loc + tidy*6*size]          =  4*Atd[tidy][0][tidx] - 5*Atd[tidy][2][tidx] +   Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + size]   = -4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] +   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 2*size] =  4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] -   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 3*size] = -2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] + 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 4*size] =  2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] - 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 5*size] =  4*Atd[tidy][1][tidx] - 5*Atd[tidy][3][tidx] +   Atd[tidy][5][tidx];
    __syncthreads();
}