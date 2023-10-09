// #pragma once
#include "util.h"
class testCase{
public:
    int inside;
    int chn;
    int bat4Conv;
    int nInput;
    float *input_cpu;
    float *input_gpu;  //, *output_gpu;
    
    char inputname[60];

    testCase(char* fileName, int size, int batch, int channel){
        strcpy(inputname, fileName);
        inside = size;
        bat4Conv = batch;
        chn = channel;
        
        nInput = bat4Conv * inside * inside * (chn) +1;
        input_cpu = get_parameter(inputname, nInput);
        
        cudaMalloc((void **) &input_gpu, nInput<<2);
        cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
    }
};