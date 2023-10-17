#include "util.h"

class testCase{
    public:
    int index;
    int bat4Conv;
    int bat4Gemm = 36;
    int inside, inside_beta;
    int blockn;
    int numOfFilter;
    int M,N;
    int MSize, NSize;
    int oside;
    int nGemmOutput;

    char gemmOutputName[60];

    float *gemmOutput_cpu;
    float *gemmOutput_gpu;

    testCase(char* fileName, int size, int batch, int channel, int channel_NextLevel, int idx){
        index = idx;
        
        char str2[5];
        char str3[20] = "/M3_new0.bin";
        sprintf(str2,"%d", index);
        strcpy(gemmOutputName, fileName);
        strcat(gemmOutputName, str2);
        strcat(gemmOutputName, str3);

        numOfFilter = channel_NextLevel;
        bat4Conv = batch;
        // chn = channel;
        int padding =1;
        int marginOfInputSide = (size+2*padding-6)%4;
        bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
        inside_beta = sideCheck? size +2*padding : (size + 2*padding + 4 - marginOfInputSide);
        blockn = (inside_beta -2) /4;
        M = bat4Conv * blockn * blockn;
        N = channel_NextLevel;

        bool MCheck = ( M % 128 == 0 )? true: false;
        bool NCheck = ( N % 128 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        NSize = NCheck? N: ((N/128 +1) * 128);

        oside = size +2*padding - 2;
        
        nGemmOutput = 36* MSize *NSize;

        // printf("MSize:%d, NSize:%d \n", MSize, NSize);
        gemmOutput_cpu = get_parameter(gemmOutputName, nGemmOutput);

        cudaMalloc((void**) &gemmOutput_gpu, nGemmOutput<<2);

        cudaMemcpy(gemmOutput_gpu, gemmOutput_cpu, nGemmOutput<<2, cudaMemcpyHostToDevice);
    }
};