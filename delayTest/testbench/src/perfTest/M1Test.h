// #pragma once
#include "testCase.h"
// #include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new1(int side, int side_beta, int MSize, int KSize, int padding, int blockn_beta, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new2(int side, int side_beta, int MSize, int KSize, int padding, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, int blockn_beta, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new4(int side, int side_beta, int MSize, int KSize, int padding, int blockn_beta, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new5(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, int blockn_beta, float * pinputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new6(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, float * pinputs, float* pOutputs);

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

    int marginOfInputSide, M, K;
    bool sideCheck, MCheck, KCheck;

    float* inputTran_gpu;
    float* inputTran_cpu;
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
        minSide = 4;
        maxSide = 1024;
        maxChannel = 512;
        maxBatch = 1024;
        // printf("baseClasse default constructor\n minSide:%d, maxSide:%d maxChannel:%d maxBatch:%d\n",minSide, maxSide, maxChannel, maxBatch);
    }
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
    }   else {
        printf("Valid False\n");
        return false;
    }
}

float inputTransMethod::testPerformance(testCase tc){
    
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
    
    cudaMemcpy(inputTran_cpu, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    printf("first element:%f\n",inputTran_cpu[0]);
    save_parameter(outFileName, nInputTran, inputTran_cpu);

    return singleTime;
    // profile the timing of the program
    
    // avetime = avetime/10;
}

class orig : public inputTransMethod{
    public:
    orig(){
        maxBatch =1024;
        minSide = 6;
        maxSide = 262142;
        maxChannel = 512000;
        strcpy(outFileName, "./data/M1_orig.bin");
    }
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128<<<dim3(tc.chn,blockn,blockn),dim3(tc.bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new1 : public inputTransMethod{
    public:
    new1(){
        minSide = 34;
        maxSide = 69632;
        maxBatch = 65535;
        maxChannel = 512000;
        strcpy(outFileName, "./data/M1_new1.bin");
    }
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        printf("blockn:%d, blockn_beta:%d\n",blockn_makeup,blockn);
        wino_input_trans_nchw_suitFor128_new1<<<dim3(tc.chn,(blockn_makeup*blockn_makeup/64),tc.bat4Conv),dim3(8,8)>>>(tc.inside, inside_beta, MSize, KSize, padding, (blockn_makeup/8), tc.input_gpu, inputTran_gpu);
    }
};

class new2 : public inputTransMethod{
    public:
    new2(){
        maxChannel = 1024;
        minSide = 34;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(outFileName, "./data/M1_new2.bin");
    } 
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128_new2<<<dim3(blockn,blockn,tc.bat4Conv),dim3(tc.chn)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
    }
};

class new3 : public inputTransMethod{
    public:
    new3(){
        maxChannel = 65535;
        minSide = 34;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(outFileName, "./data/M1_new3.bin");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        int chn_makeup = (tc.chn%4)? (tc.chn+4-tc.chn%4):tc.chn;
        wino_input_trans_nchw_suitFor128_new3<<<dim3(tc.bat4Conv,(blockn_makeup*blockn_makeup/64),chn_makeup/4),dim3(8,8,4)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.chn, blockn, (blockn_makeup/8), tc.input_gpu, inputTran_gpu);
    }
};

class new4 : public inputTransMethod{
    public:
    new4(){
        maxChannel = 65535;
        minSide = 34;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(outFileName, "./data/M1_new4.bin");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        wino_input_trans_nchw_suitFor128_new4<<<dim3(tc.chn,(blockn_makeup*blockn_makeup/64),tc.bat4Conv),dim3(8,8)>>>(tc.inside, inside_beta, MSize, KSize, padding, (blockn_makeup/8), tc.input_gpu, inputTran_gpu);
    }
};

class new5 : public inputTransMethod{
    public:
    new5(){
        maxChannel = 65535;
        minSide = 34;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(outFileName, "./data/M1_new5.bin");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%8) ? (blockn+8 - blockn%8): blockn;
        // printf("blockn:%d, blockn_makeup:%d\n", blockn, blockn_makeup);
        int chn_makeup = (tc.chn%4)? (tc.chn+4-tc.chn%4):tc.chn;
        wino_input_trans_nchw_suitFor128_new5<<<dim3(tc.bat4Conv,(blockn_makeup*blockn_makeup/64),chn_makeup/4),dim3(8,8,4)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.chn, blockn, (blockn_makeup/8), tc.input_gpu, inputTran_gpu);
    }
};

class new6 : public inputTransMethod{
    public:
    new6(){
        maxChannel = 65535;
        minSide = 18;
        maxSide = 65535;
        maxBatch = 65535;
        strcpy(outFileName, "./data/M1_new6.bin");
    } 
    virtual void execut(testCase tc) {
        int blockn_makeup= (blockn%4) ? (blockn+4 - blockn%4): blockn;
        // printf("blockn:%d, blockn_makeup:%d\n", blockn, blockn_makeup);
        int chn_makeup = (tc.chn%4)? (tc.chn+4-tc.chn%4):tc.chn;
        wino_input_trans_nchw_suitFor128_new6<<<dim3(tc.bat4Conv,(blockn_makeup*blockn_makeup/16),chn_makeup/4),dim3(4,4,4)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.chn, blockn, tc.input_gpu, inputTran_gpu);
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

// find the parallalism in spatial dims 
__global__ void wino_input_trans_nchw_suitFor128_new1(int side, int side_beta, int MSize, int KSize, int padding, int blockn_beta,float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-8
    int tidy = threadIdx.y;  // 0-8
    int bidx = blockIdx.x;  // chn
    int bidy = blockIdx.y;  // blockn by 64
    int bidz = blockIdx.z;  // bat4Conv

    int blockn = 1+(side_beta -6)/4;
    int totalChn = gridDim.x;
    int numOfBatch = gridDim.z;

    // extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};

    int blocknX = bidy % ((blockn+7)/8);
    int blocknY = bidy / ((blockn+7)/8);


    if (blocknX == blockn_beta-1 || blocknY ==blockn_beta -1) {
        if ((blocknX*8 + tidx >=blockn) || (blocknY*8 + tidy>=blockn)){
            return;
        }  
    }

    if(blocknX == 3 && tidx == 4 && blocknY==0 && tidy ==0 && bidx ==0){
        int idx = bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
        bool judge1 = blocknX*8 + tidx >=blockn;
        bool judge2 = blocknY*8 + tidy>=blockn;
        printf("idx:%d  idy:%d verify:%d, %d\n",blocknX*8 + tidx, blocknY*8+tidy, judge1, judge2);

    }

    if(blocknX == 3 && tidx == 4 && blocknY==1 && tidy ==0 && bidx ==0){
        int idx = bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
        bool judge1 = blocknX*8 + tidx >=blockn;
        bool judge2 = blocknY*8 + tidy>=blockn;
        printf("idx:%d  idy:%d verify:%d, %d\n",blocknX*8 + tidx, blocknY*8+tidy, judge1, judge2);
        printf("gridDim.y;%d, blockn:%d\n",gridDim.y,blockn);
        printf("prepare for bool test\n blocknX:%d, blockn_beta:%d\n",blocknX,blockn_beta);
        if (blocknX == blockn_beta-1 || blocknY ==blockn_beta-1) {
            printf("test1\n");
            if ((blocknX*8 + tidx >=blockn) || (blocknY*8 + tidy>=blockn)){
                printf("test2\n");
            }  
        }
    }

    int inside = side;
    
    pInputs = &pInputs[tidx*4 + tidy*4*inside + bidx*inside*inside + bidz*totalChn*inside*inside + blocknX*32 + blocknY*32*inside - padding*(1+inside)];
    pOutputs= &pOutputs[bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize ];


    // for(){
    //     // store the data into the shared cache
    //     // constrain the range of the input
    //     inCache = (pInputs + );
    // }

    // load the data from the shared into the private
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            if( (4*tidx + j + blocknX*32 >=padding ) && (4*tidy + i + blocknY*32 >= padding) && 
                (4*tidx + j + blocknX*32 <= inside -1 + padding ) && (4*tidy + i + blocknY*32 <= inside -1 + padding)){
                Mread[i][j] = pInputs[j+i*inside];
            }
        }
    }

    // do calculation of each thread;
    // each thread calculate for 
    // each block claculate for
    float Atd[6][6] = {{0}};
    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    // store the result back to the output matrix;
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

    if(blocknX == 0 && tidx == 0 && blocknY==1 && tidy ==1 && bidx ==0){
        int idx = bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
        printf("idx:%d\n",idx);
        // for(int i=0;i<6;i++){
            
        //     printf("%f ",Mread[i][0]);
        //     printf("%f ",Mread[i][1]);
        //     printf("%f ",Mread[i][2]);
        //     printf("%f ",Mread[i][3]);
        //     printf("%f ",Mread[i][4]);
        //     printf("%f \n",Mread[i][5]);
        // }

        // for(int i =0;i<6;i++) {
        //     printf("%f ",4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4] );
        //     printf("%f ",-4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4]  );
        //     printf("%f ",4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4]); 
        //     printf("%f ",-2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4]); 
        //     printf("%f ",2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4]); 
        //     printf("%f \n",4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5]); 
        // }
    }
    // if(blocknX == 3 && tidx == 4 && blocknY==1 && tidy ==0 && bidx ==0){
    //     int idx = bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
    //     printf("idx:%d\n",idx);
    // }
}

// find the parallalism in channel dims
// TODO: need to think about the computation efficiency. 
// And the access consistency of banks of the loading data. 
// latency hidden as well. Does it need a shift of the input? 
// wino_input_trans_nchw_suitFor128_new2<<<dim3(bat4Conv, blockn, blockn),dim3(chn,1,1), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new2(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tid = threadIdx.x;  // 0-1024, chn_in
    
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    // if (tidx ==0 && tidy==0 && tidz==0 && bid ==0)
    //     printf("(bat4Conv,blockn,blockn,chn_in)=(%d,%d,%d,%d)\n",blockDim.z,blockDim.x,blockDim.y,gridDim.x);
    // printf("input_gpu[0]:%lf\n",pInputs[0]);

    int totalChn = blockDim.x;
    int numOfBatch = gridDim.z;
    int blockn = gridDim.x;

    // extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};

    int inside = side;

    pInputs = &pInputs[bidx*4 + bidy*4*inside + tid*inside*inside + bidz*totalChn*inside*inside - padding*(1+inside) ];
    pOutputs = &pOutputs[bidz + bidx*numOfBatch + bidy*numOfBatch*blockn + tid*MSize];

    for(int i=0;i<6;i++) 
    {
        for(int j=0; j<6; j++)
        {
            if( ( 4*bidx+j >= padding )&&( 4*bidy+i >= padding )&&( 4*bidx+j<=inside-1+padding)&&(4*bidy+i<=inside-1+padding) )
            {
                Mread[i][j] = pInputs[j+i*inside];
            }
        }
    }

    float Atd[6][6] ={{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = MSize *KSize;

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

// find the parallalism both in channel dims and spatial dims 
// wino_input_trans_nchw_suitFor128_new3<<<dim3(bat4Conv, blockn*blockn/(64), chn/4),dim3(8,8,4), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, int blockn_beta ,float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-8,
    int tidy = threadIdx.y;  // 0-8
    int tidz = threadIdx.z;  // 0-4
    int bidx = blockIdx.x;   // batch
    int bidy = blockIdx.y;   // blockn by 64
    int bidz = blockIdx.z;   // chn by 4

    int totalChn = numOfChn;
    int numOfBatch = gridDim.x;
    int blockn = numOfBlockn;
    // int blockn = 1+(inside -6)/4;

    int blocknX = bidy % ((blockn+7)/8);
    int blocknY = bidy / ((blockn+7)/8);

    if (blocknX == blockn_beta-1 || blocknY ==blockn_beta -1) {
        if ((blocknX*8 + tidx >=blockn) || (blocknY*8 + tidy>=blockn)){
            return;
        }  
    }

    if(tidz+bidz*4 >= numOfChn){
        return;
    }

    float Mread[6][6] = {{0}};

    int inside = side;

    

    // if(tidx ==0 && tidy==0 && bidx==0 && bidy==0 && bidz ==0 ){
    //     printf("inside:%d inside_beta:%d totalChn:%d numOfBatch:%d blockn%d totalLargeBlockn:%d \n",inside, side_beta, totalChn, numOfBatch, blockn, gridDim.y);
    // }


    pInputs = &pInputs[tidx*4 +tidy*4*inside + tidz*inside*inside + bidx*inside*inside*totalChn+ blocknX*32 +blocknY*32*inside + bidz*inside*inside*4 - padding*(1+inside)];

    pOutputs= &pOutputs[bidx + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*numOfBatch*blockn + blocknY*numOfBatch*blockn*8 + tidz*MSize + bidz*MSize*4];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*tidx + j + blocknX*32 >= padding) && ( 4*tidy +i + blocknY*32 >= padding) &&
                (4*tidx + j + blocknX*32 <=inside-1+padding) && (4*tidy + i + blocknY*32 <= inside-1 +padding)) {
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

// based on the new1 and complete the preload data into the shared cache
__global__ void wino_input_trans_nchw_suitFor128_new4(int side, int side_beta, int MSize, int KSize, int padding, int blockn_beta, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-8
    int tidy = threadIdx.y;  // 0-8
    int bidx = blockIdx.x;  // chn
    int bidy = blockIdx.y;  // blockn by 64
    int bidz = blockIdx.z;  // bat4Conv

    int inside = side;

    int blockn = (side_beta -2)/4;
    int totalChn = gridDim.x;
    int numOfBatch = gridDim.z;

    int blocknX = bidy % ((blockn+7)/8);
    int blocknY = bidy / ((blockn+7)/8);

    // __shared__ float inCache[34 * 34] = {0};
    __shared__ float inCache[34 * 34]; // 4 * 8 + 2 = 34 elements in each edge. WARNING: not initialized
    float Mread[6][6] = {{0}};

    // pInputs = &pInputs[tidx*4 + tidy*4*inside + bidx*inside*inside + bidz*totalChn*inside*inside + blocknX*32 + blocknY*32*inside - padding*(1+inside)]; 
    pInputs = &pInputs[bidx*inside*inside + bidz*totalChn*inside*inside + blocknX*32 + blocknY*32*inside - padding*(1+inside)];
    pOutputs= &pOutputs[bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize ];


    //load the 8x8 tile(input) to shared
    int tid = tidx + 8*tidy; // 0-63

    for(int cnt =0; cnt<19; cnt++) {
        int loadx = (cnt*64+tid)%34;
        int loady = (cnt*64+tid)/34;
        bool rangeVerify = (blocknX*32 + loadx >= padding) && (blocknY*32+loady) >= padding && 
            (blocknX*32 + loadx <=inside-1+padding) && (loady+blocknY*32<=inside-1+padding);
        
        inCache[cnt*64+tid] = rangeVerify? pInputs[loadx+loady*inside] : 0;
        
    }

    __syncthreads();

    if (blocknX == blockn_beta-1 || blocknY ==blockn_beta -1) {
        if ((blocknX*8 + tidx >=blockn) || (blocknY*8 + tidy>=blockn)){
            return;
        }  
    }

    // load the data from the shared into the private
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            // if( (4*tidx + j + blocknX*32 >=padding ) && (4*tidy + i + blocknY*32 >= padding) && 
            //     (4*tidx + j + blocknX*32 <= inside -1 + padding ) && (4*tidy + i + blocknY*32 <= inside -1 + padding)){
            //     Mread[i][j] = pInputs[j+i*inside];
            // }
            Mread[i][j] = inCache[(tidy*4+i)*34 + tidx*4+j];
        }
    }


    //****************************
    // DEBUG
    // if(tidx ==0 && tidy ==4 && bidx ==0 && bidy ==14 && bidz ==0 ){
    //     // int idx =bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
    //     // printf("blockX:%d, blockY:%d, place:%d\n",blocknX, blocknY, idx);
    //     for(int i=0; i<6;i++){
    //         for(int j=0;j<6;j++){
    //             printf("%f  ", Mread[i][j]);
    //         }printf("\n");
    //         // printf("diff: %d\n", 4*tidx);   
    //     }
    // }

    // __syncthreads();
    
    // if(tidx ==0 && tidy ==4 && bidx ==0 && bidy ==14 && bidz ==0 ){
    //     printf("\n\n\n");
    //     // int idx =bidz + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*blockn*numOfBatch + blocknY*blockn*numOfBatch*8 + bidx*MSize;
    //     // printf("blockX:%d, blockY:%d, place:%d\n",blocknX, blocknY, idx);
    //     for(int i=0; i<6;i++){
    //         for(int j=0;j<6;j++){
    //             printf("%f  ", Mread[i][j]);
    //         }printf("\n");
    //     }
    // }
    //**************************
    

    // do calculation of each thread;
    // each thread calculate for 
    // each block claculate for
    float Atd[6][6] = {{0}};
    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    // store the result back to the output matrix;
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

// based on the new3 and complete the preload data into the shared cache
__global__ void wino_input_trans_nchw_suitFor128_new5(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, int blockn_beta, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-8,
    int tidy = threadIdx.y;  // 0-8
    int tidz = threadIdx.z;  // 0-4
    int bidx = blockIdx.x;   // batch
    int bidy = blockIdx.y;   // blockn by 64
    int bidz = blockIdx.z;   // chn by 4

    int inside = side;
    int totalChn = numOfChn;
    int numOfBatch = gridDim.x;
    int blockn = numOfBlockn;
    // int blockn = 1+(inside -6)/4;

    int blocknX = bidy % ((blockn+7)/8);
    int blocknY = bidy / ((blockn+7)/8);

    if (blocknX == blockn_beta-1 || blocknY ==blockn_beta -1) {
        if ((blocknX*8 + tidx >=blockn) || (blocknY*8 + tidy>=blockn)){
            return;
        }  
    }

    if(tidz+bidz*4 >= numOfChn){
        return;
    }

    // extern __shared__ float inCache[2][x][i dont know];
    float Mread[6][6] = {{0}};

    pInputs = &pInputs[tidx*4 +tidy*4*inside + tidz*inside*inside + bidx*inside*inside*totalChn+ blocknX*32 +blocknY*32*inside + bidz*inside*inside*4 - padding*(1+inside)];
    pOutputs= &pOutputs[bidx + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*numOfBatch*blockn + blocknY*numOfBatch*blockn*8 + tidz*MSize + bidz*MSize*4];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*tidx + j + blocknX*32 >= padding)&&( 4*tidy +i + blocknY*32 >= padding)&&
                (4*tidx + j + blocknX*32 <=inside-1+padding) && (4*tidy + i + blocknY*32 <= inside-1 +padding)) {
                Mread[i][j] = pInputs[j + i * inside];
            }

            //TODO: load from inCache
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

// find the parallalism both in channel dims and spatial dims & preload data into the shared cache
// wino_input_trans_nchw_suitFor128_new6<<<dim3(bat4Conv, blockn*blockn/(16), chn/4),dim3(4,4,4), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new6(int side, int side_beta, int MSize, int KSize, int padding, int numOfChn, int numOfBlockn, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-4,
    int tidy = threadIdx.y;  // 0-4
    int tidz = threadIdx.z;  // 0-4
    int bidx = blockIdx.x;   // batch
    int bidy = blockIdx.y;   // blockn by 16
    int bidz = blockIdx.z;   // chn by 4

    if(tidz+bidz*4 >= numOfChn){
        return;
    }

    int totalChn = numOfChn;
    int numOfBatch = gridDim.x;
    int blockn = numOfBlockn;
    // int blockn = 1+(inside -6)/4;

    float Mread[6][6] = {{0}};

    int inside = side;

    int blocknX = bidy % ((blockn+3)/4);
    int blocknY = bidy / ((blockn+3)/4);

    pInputs = &pInputs[tidx*4 +tidy*4*inside + tidz*inside*inside + bidx*inside*inside*totalChn+ blocknX*16 +blocknY*16*inside + bidz*inside*inside*4 - padding*(1+inside)];

    pOutputs= &pOutputs[bidx + tidx*numOfBatch + blocknX*4*numOfBatch + tidy*numOfBatch*blockn + blocknY*numOfBatch*blockn*4 + tidz*MSize + bidz*MSize*4];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*tidx + j + blocknX*16 >= padding)&&( 4*tidy +i + blocknY*16 >= padding)&&
                (4*tidx + j + blocknX*16 <=inside-1+padding) && (4*tidy + i + blocknY*16 <= inside-1 +padding)) {
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