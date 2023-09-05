// This file is used for recap after a long time shifting from other project.
// This is the v1.0 recap, do not including padding and edging processing.


#include "recaplib_v1_0.h"

int main(int argc, char argv[]){
    char *inputName = "../data/input.bin";
    char *kernelName = "../data/kernel.bin";
    int bat4conv = atoi(argv[1]);       //16
    int size = atoi(argv[2]);           //22
    int chn = atoi(argv[3]);            //32
    int numOfFilter = atoi(argv[4]);    //128

    int nInput = bat4conv* size*size *chn;
    int nKernel = numOfFilter* 3*3*chn;
    int nOutput =  bat4conv * (size-2)* (size-2) * numOfFilter;

    float *input_cpu = get_parameter(inputName);
    float *kernel_cpu = get_parameter(kernelName);
    float *input_gpu;
    float *kernel_gpu;

    cudaMalloc((void **) &input_gpu, nInput<<2);
    cudaMalloc((void **) &kernel_gpu, nKernel<<2);
    cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_gpu, kernel_cpu, nKernel<<2, cudaMemcpyHostToDevice);

//version one cudnn
{
    
}

//version two self-defined cuda
{
    float *inputTran_gpu;
    float *kernelTran_gpu;
    float *gemmOutput_gpu;
    float *output_gpu;

    Wino_inputTran_NHWC<<<()>>>();
    Wino_KernelTran_NHWC<<<>>>();
    GEMM<<<>>>();
    Wino_inver_NHWC<<<>>>();

    Malloc((void **) output_cpu, nOutput<<2);
    cudaMalloc((void **) output_gpu, nOutput<<2);
    cudaMemcpy(output_cpu, output_gpu, nOutput<<2, cudaMemcpyDeviceToHost);
    save_parameter(outputName, output_cpu);
}

}

