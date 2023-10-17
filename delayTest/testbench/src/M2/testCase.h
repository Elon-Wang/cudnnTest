#include "util.h"

class testCase{
public:
    int numOfFilter;
    int chn;
    int nKernel;
    int index;
    float *kernel_cpu;
    float *kernel_gpu;

    char kernelName[60];

    testCase(char* fileName, int size, int batch, int channel, int channel_NextLevel, int idx){
        numOfFilter = channel_NextLevel;
        chn = channel;
        strcpy(kernelName, fileName);
        index = idx;

        nKernel = numOfFilter * 3*3 * chn;
        kernel_cpu = get_parameter(kernelName, nKernel);
        
        cudaMalloc((void **) &kernel_gpu, nKernel<<2);
        cudaMemcpy(kernel_gpu, kernel_cpu, nKernel<<2, cudaMemcpyHostToDevice);
    }
};