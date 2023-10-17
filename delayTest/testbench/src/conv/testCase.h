#include "util.h"

class testCase{
    public:
    int index;
    int side, chn, bat4Conv, numOfFilter;
    int nInput, nKernel;
    float* input_cpu;
    float* input_gpu;
    float* kernel_cpu;
    float* kernel_gpu;
    // char outputFile[60];

    char inputName[60];
    char kernelName[60];

    testCase(char* inputFile, char* kernelFile, int size, int batch, int chn_in, int chn_out,int idx){
        index = idx; 
        bat4Conv = batch;
        chn = chn_in;
        numOfFilter = chn_out;
        side = size;

        char str2[5];
        char input_str3[20] = "/input.bin";
        char kernel_str3[20] = "/kernel.bin";
        sprintf(str2, "%d", index);

        strcpy(inputName, inputFile);
        // strcat(inputName, str2);
        strcat(inputName, input_str3);

        strcpy(kernelName, kernelFile);
        // strcat(kernelName, str2);
        strcat(kernelName, kernel_str3);

        nInput = bat4Conv * side *side * chn;
        nKernel= numOfFilter *3*3* chn;

        input_cpu = get_parameter(inputName, nInput);
        kernel_cpu = get_parameter(kernelName, nKernel);

        cudaMalloc((void **) &input_gpu, nInput<<2);
        cudaMalloc((void **) &kernel_gpu, nKernel<<2);
        
        cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_gpu, kernel_cpu, nKernel<<2, cudaMemcpyHostToDevice);
    }
};