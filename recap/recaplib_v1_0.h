#include <cuda.h>

__global__ void Wino_inputTran_NHWC(float *input, float* inputTran, int bat4conv, int size, int chn){
    int tid = threadIdx.x;  // channel
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    float Mbuffer[6][6];

    float *pInput = (input + bidx*4*chn + bidy*4*chn*size + bidz*size*size*chn);
    
    #unloop
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            Mbuffer[i][j] = pInput + j + i *size + tid;
        }
    }

    
}

__global__ void Wino_KernelTran_NHWC(float *kernel, float* kernelTran, int numOfFilter, int size, int chn){

}

__global__ void GEMM(float* A, float* B, float* C, int m, int n, int k){

}

__global__ void Wino_inver_NHWC(float* input, float* output, int bat4Conv, int size, int numOfFilter){

}
