#include <cuda.h>

__global__ void Wino_inputTran_NHWC(float *input, float* inputTran, int bat4conv, int size, int chn){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float Mbuffer[6][6];
    
}

__global__ void Wino_KernelTran_NHWC(float *kernel, float* kernelTran, int numOfFilter, int size, int chn){

}

__global__ void GEMM(float* A, float* B, float* C, int m, int n, int k){

}

__global__ void Wino_inver_NHWC(float* input, float* output, int bat4Conv, int size, int numOfFilter){

}
