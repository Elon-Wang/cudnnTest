#include "GEMM.cuh"

__global__ void warmup(){}
void wrapedConv_NHWC(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float **output);
// void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float **output);
__global__ void wino_input_trans_nhwc_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs );
__global__ void wino_kernel_trans_nhwc_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);
__global__ void wino_invers_nhwc_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);

void wrapedConv_NHWC(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
    // int nInput = 64*128*128;
    // float *output1 =(float *)malloc(nInput * sizeof(float));
    // cudaMemcpy(output1, m1, nInput<<2, cudaMemcpyDeviceToHost);
    // printf("1st:%f\n",output1[0]);

    int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);
    
    int blockn = (inside_beta -2) /4;
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;

    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    int blockx, blocky, bat4Gemm;

    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm =36;
    
    int oside = inside + 2*padding -2;
    // int nInputTran = 36*MSize*KSize;
    // int nFilterTran = 36*MSize*KSize;

    // printf("blockn, bat4Conv, chn = %d %d %d\n inside, inside_beta, MSize, KSize = %d,%d,%d,%d\n",blockn, bat4Conv, chn, inside, inside_beta,MSize,KSize);

    // wino_input_trans_nhwc_suitFor128<<< dim3(1, 1, 1), dim3(1,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    wino_input_trans_nhwc_suitFor128<<< dim3(blockn, blockn, bat4Conv), dim3(chn,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
    
    // float *output2 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output2, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // printf("2st:%f\n",output2[0]);

    wino_kernel_trans_nhwc_suitFor128<<< dim3(numOfFilter,1,1), dim3(chn,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // wino_kernel_trans_nhwc_suitFor128<<< dim3(1,1,1), dim3(1,1,1)>>>(NSize, KSize, m2, filterTran_gpu); 
    // float *output3 =(float *)malloc(nFilterTran * sizeof(float));
    // cudaMemcpy(output3, filterTran_gpu, nFilterTran<<2, cudaMemcpyDeviceToHost);
    // printf("3st:%f\n",output3[0]);
    
    // GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    //TODO:
    GEMM_batch_256_128x128_MKNK<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize, NSize, KSize, 1, inputTran_gpu, filterTran_gpu, 0, gemmOutput_gpu);
    // int nGemmOutput = 36*MSize*NSize;
    // float *output1 =(float *)malloc(nGemmOutput * sizeof(float));
    // cudaMemcpy(output1, gemmOutput_gpu, nGemmOutput<<2, cudaMemcpyDeviceToHost);
    // printf("1st:%f\n",output1[0]);
    wino_invers_nhwc_suitFor128<<< dim3(bat4Conv, blockn, blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);


    chn = numOfFilter;

    cudaDeviceSynchronize();
    
}
/*
    In this design, one thread coresponding to a tile, and the threads inside a block corresponding to each channel of a batch
    The block config should be (blockx, blocky, batch).
    The M-dim of the GEMM should be arranged as blockx * blocky * bat4Conv, bat4Conv is the last dimension.
*/
// change to MKNK output
__global__ void wino_input_trans_nhwc_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // chn_in
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // batch

    int totalChn = blockDim.x;
    int blockn = gridDim.x;
    int numOfBatch = gridDim.z;

    float Mread[6][6] = {{0}};

    int inside = side;

    // take care of x-direction and y direction.
    pInputs = &pInputs[tidx + bidx*4*totalChn +bidy*4*inside*totalChn + bidz*inside*inside*totalChn - padding*(padding+inside)*totalChn ];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidz*KSize + bidx*numOfBatch*KSize + bidy*numOfBatch*blockn*KSize];
    // pOutputs = &pOutputs[tidx + bidx*numOfBatch + bidy*numOfBatch*blockn + bidx*MSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j * totalChn + i * inside * totalChn];
            }
        }
    }

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

// change to MKNK output
__global__ void wino_kernel_trans_nhwc_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // chn_in
    int bid  = blockIdx.x;   // numOfFilter

    int totalChn = blockDim.x;


    float Mread[3][3] ={{0}};
    pInputs = &pInputs[tidx + bid *3*3*gridDim.x ];
    pOutputs = &pOutputs[tidx + bid*KSize];

    #pragma unroll
    for (int i= 0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Mread[i][j] = pInputs[j * totalChn + i * 3 * totalChn];
        }
    }

    float Gg[6][3] = {{0}};
    
    for(int i=0;i<3;i++){
        Gg[0][i] = Mread[0][i]/4;
        Gg[1][i] = -Mread[0][i]/6 - Mread[1][i]/6 - Mread[2][i]/6;
        Gg[2][i] = -Mread[0][i]/6 + Mread[1][i]/6 - Mread[2][i]/6;
        Gg[3][i] = Mread[0][i]/24 + Mread[1][i]/12 + Mread[2][i]/6;
        Gg[4][i] = Mread[0][i]/24 - Mread[1][i]/12 + Mread[2][i]/6;
        Gg[5][i] = Mread[2][i];
    }

    int size = NSize * KSize;

    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = Gg[i][0]/4;
        pOutputs[i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 5*size] = Gg[i][2];
    }
}

__global__ void wino_invers_nhwc_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int chn = threadIdx.x; // numOfFilter or chn_out
    int bidx = blockIdx.x; // bat4Conv 
    int bidy = blockIdx.y; // blockn.x
    int bidz = blockIdx.z; // blockn,y

    float Mread[6][6] = {{0}};
    //take care, this should be the total MSize and NSize, rather than M and N;
    int size = MSize * NSize;
    int chn_out = blockDim.x;
    int bat4Conv = gridDim.x;
    int blockn = gridDim.y;

    pInputs = &pInputs[chn + bidx* NSize + bidy* bat4Conv*NSize + bidz*bat4Conv*NSize*blockn];
    // pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];
    pOutputs = &pOutputs[chn + bidx*oside*oside*chn_out +bidy*chn_out*4 + bidz*chn_out*oside*4];

    for (int i=0;i<6;i++) {
        for (int j=0; j<6; j++) {
            Mread[i][j] = pInputs[j*size + i*6*size];
        }
    }

    //output
    float Atd[4][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
        Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
        Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
    }

    // float cache[4][4] = {{0}};
    for(int i=0;i<4;i++){
        Mread[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        Mread[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        Mread[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        Mread[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        // pOutputs[i*oside*bat4Conv] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        // pOutputs[i*oside*bat4Conv + bat4Conv] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 2*bat4Conv] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        // pOutputs[i*oside*bat4Conv + 3*bat4Conv] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*bidy+j<=oside-1 ) && ( 4*bidz+i<=oside-1 ) ){
                pOutputs[i*oside*chn_out + j*chn_out] = Mread[i][j];
            }
        }
    }
}