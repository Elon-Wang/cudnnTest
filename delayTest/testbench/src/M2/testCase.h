#include "util.h"

class testCase{
public:
    int numOfFilter;
    int chn;
    int nKernel;
    float *kernel_cpu;
    float *kernel_gpu;

    char kernelName[60];

    testCase(char* fileName, int batch, int channel ){
        numOfFilter = batch;
        chn = channel;
        strcpy(kernelName, fileName);

        nKernel = numOfFilter * 3*3 * chn;
        kernel_cpu = get_parameter(kernelName, nKernel);
        
        cudaMalloc((void **) &kernel_gpu, nKernel<<2);
        cudaMemcpy(kernel_gpu, kernel_cpu, nKernel, cudaMemcpyHostToDevice);
    }
}