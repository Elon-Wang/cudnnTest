#include <cuda.h>

__global__ void Wino_inputTran_NHWC(float *pInput, float* pOutput, int bat4conv, int size, int chn){
    int tid = threadIdx.x;  // channel
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int blockn = gridDim.x;
    int Msize = blockn*blockn* bat4conv;
    int Ksize = chn;

    pInput = (pInput + bidx*4*chn + bidy*4*chn*size + bidz*size*size*chn);
    pOutput = pOutput + bidx + bidy*blockn + bidz + tid* Msize;
    
    float Mread[6][6];
    #unloop
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            Mread[i][j] = pInput + j + i *size + tid;
        }
    }

    float Btd[6][6];
    for (int i=0; i<6; i++) {
        Btd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Btd[1][i] = -4*Mread[1][i] - 4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Btd[2][i] = 4*Mread[1][i] - 4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Btd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Btd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Btd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = Msize * Ksize;
    for(int i=0; i<6; i++) {
        pOutput[i*6*size ] = 4*Btd[i][0] -5*Btd[i][2] + Btd[i][4]; 
        pOutput[i*6*size + size] = -4*Btd[i][1] -4*Btd[i][2] + Btd[i][3] + Btd[i][4];
        pOutput[i*6*size + 2*size] = 4*Btd[i][1] -4*Btd[i][2] - Btd[i][3] + Btd[i][4];
        pOutput[i*6*size + 3*size] = -2*Btd[i][1] -Btd[i][2] + 2*Btd[i][3] + Btd[i][4];
        pOutput[i*6*size + 4*size] = 2*Btd[i][1] -Btd[i][2] - 2*Btd[i][3] + Btd[i][4];
        pOutput[i*6*size + 5*size] = 4*Btd[i][1] -5*Btd[i][3] + Btd[i][5];
    }
}

__global__ void Wino_KernelTran_NHWC(float *pInput, float* pOutput, int numOfFilter, int size, int chn){
    int tid = threadIdx.x;  //chn
    int bid = blockIdx.x;   //numOfFilter

    float* Mread[3][3];

    pInput = pInput + tid + bid*3*3*chn;
    pOutput = pOutput + bid*chn +chn;

    int Ksize = chn;
    int Nsize = numOfFilter;
    int size = Ksize * Nsize;

    #pragma unloop
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            Mread[i][j] = *(pInput +j*chn + i*3*chn);
        }
    }

    float* Gg[6][6];

    for (int i=0; i<3; i++){
        Gg[0][i] = Mread[0][i]/4;
        Gg[1][i] = -Mread[0][i]/6 - Mread[1][i]/6 - Mread[2][i]/6;
        Gg[2][i] = -Mread[0][i]/6 + Mread[1][i]/6 - Mread[2][i]/6;
        Gg[3][i] = Mread[0][i]/24 + Mread[1][i]/12 + Mread[2][i]/6;
        Gg[4][i] = Mread[0][i]/24 - Mread[1][i]/12 + Mread[2][i]/6;
        Gg[5][i] = Mread[2][i];
    }

    for(int i=0; i<6; i++){
        pOutput[size*6*i] = Gg[i][0]/4; 
        pOutput[size*6*i + size] = -Gg[i][0]/6 - Gg[i][1]/6 - Gg[i][2]/6;
        pOutput[size*6*i + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 - Gg[i][2]/6;
        pOutput[size*6*i + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutput[size*6*i + 4*size] = Gg[i][0]/24 -  Gg[i][1]/12 + Gg[i][2]/6;
        pOutput[size*6*i + 5*size] = Gg[i][2];
    }
}

//M*K K*N, matrix size 128*128
__global__ void GEMM(float* A, float* B, float* C, int m, int n, int k){
    tid = threadIdx.x;
    bidx = blockIdx.x;
    bidy = blockIdx.y;
    bidz = blockIdx.z;


}


// blockn*blockn*batch* 6*6*numOfFIlter
__global__ void Wino_inver_NHWC(float* pinput, float* poutput, int bat4Conv, int size, int numOfFilter){
    int tid = threadIdx.x;  //numOfFilter
    int bidx = blockIdx.x;  //blockn
    int bidy = blockIdx.y;  //blockn
    int bidz = blockIdx.z;  //batch

    float Mread[6][6];
    pInput = pInput + tid + bidx*6*numOfFilter + bidy *numOfFilter*6*blockn * 6 + bidz*6*6*blockn*blockn*numOfFilter;
    pOutput = pOutput + tid + bidx*4*numOfFilter + bidy*numOfFilter*4*blockn*4 + bidz*4*4*blockn*blockn*numOfFilter;

    #pragma unloop
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            Mread[i][j] = *(pInput + j*numOfFIlter + i*numOfFilter*6*blockn);
        }
    }

    float At[4][6];

    for(int i=0; i<6; i++){
        At[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
        At[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
        At[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
        At[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i]+ Mread[5][i];
    }

    for(int i=0; i<4; i++){
        pOutput[i*4*4*blockn*blockn*numOfFilter ] = At[i][0] + At[i][1] + At[i][2] + At[i][3] + At[i][4]; 
        pOutput[i*4*4*blockn*blockn*numOfFilter + numOfFilter] = At[i][1] - At[i][2] + 2*At[i][3] - 2*At[i][4];
        pOutput[i*4*4*blockn*blockn*numOfFilter + 2*numOfFilter] = At[i][1] + At[i][2] + 4*At[i][3] + 4*At[i][4];
        pOutput[i*4*4*blockn*blockn*numOfFilter + 3*numOfFilter] = At[i][1] - At[i][2] + 8*At[i][3] - 8*At[i][4] + At[i][5];
    }
}
