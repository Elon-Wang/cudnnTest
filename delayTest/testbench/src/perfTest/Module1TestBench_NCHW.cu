#include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);


int main(){

    new1 fun1();
    new2 fun2();
    new3 fun3();
    new4 fun4();

    testCase teCase[5];
    char fileName[] = "../data/input.bin";
    
    teCase[0] = new testCase(fileName, 224, 1, 1);      // size, batch, chn
    teCase[1] = new testCase(fileName, 114, 1, 64);
    teCase[2] = new testCase(fileName, 60, 1, 128);
    teCase[3] = new testCase(fileName, 30, 1, 256);
    teCase[4] = new testCase(fileName, 22, 1, 256);

    // need to test the performance of the different dims     
    
    for(int i=0; i<5; i++) {
        // test available
        // an array to get the availability;
        fun1.testValid(teCase[i]);
        fun2.testValid(teCase[i]);
        fun3.testValid(teCase[i]);
        fun4.testValid(teCase[i]);

        // performance testing
        fun1.testPerformance(teCase[i]);
        fun2.testPerformance(teCase[i]);
        fun3.testPerformance(teCase[i]);
        fun4.testPerformance(teCase[i]);

        // output the testing result

        printf("Fun1: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
        printf("Fun2: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
        printf("Fun3: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
        printf("Fun4: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
    }

}

class testCase{
    int inside;
    int chn;
    int bat4Conv;
    int nInput;
    float *input_cpu;
    float *input_gpu;  //, *output_gpu;
    
    char inputname[];

    testCase(char* fileName, int size, int batch, int channel){
        inputname[] = fileName;
        inside = size;
        bat4Conv = batch;
        chn = channel;
        
        nInput = bat4Conv * inside * inside * (chn) +1;
        *input_cpu = get_parameter(inputname, nInput);
        
        cudaMalloc((void **) &input_gpu, nInput<<2);
        cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
    }
}

class inputTransMethod{
    int minSide;
    int maxSide;
    int maxChannel;
    int maxBatch;

    // counted in micro second
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid= false;
    bool testValid(testCase tc);
    void testPerformance(testCase tc);
    virtual void execut(testCase tc) =0 ;

    inputTransMethod(){

    }

}

class new1 : public inputTransMethod{
    virtual void execut(testCase tc) {
        wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(tc.inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran1_gpu);
    }
}

class new2 : public inputTransMethod{
    
}

class new3 : public inputTransMethod{
    
}

class new4 : public inputTransMethod{
    
}

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