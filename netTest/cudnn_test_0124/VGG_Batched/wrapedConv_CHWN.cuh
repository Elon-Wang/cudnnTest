#pragma once
#include "GEMM.cuh"

#ifndef WARMUP
#define WARMUP
__global__ void warmup(){}
#endif
// void wrapedConv_CHWN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float **output);
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float **output);
__global__ void wino_input_trans_chwn_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs );
__global__ void wino_kernel_trans_chwn_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);
__global__ void wino_invers_chwn_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);
__global__ void wino_invers_chwn_suitFor128_2(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);

/*
void wrapedConv_CHWN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float* inputTran_gpu, float* filterTran_gpu, float* gemmOutput_gpu, float ** output){
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

    // cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    // cudaEventCreate(&ts0);
    // cudaEventCreate(&ts1);
    // cudaEventCreate(&ts2);
    // cudaEventCreate(&ts3);
    // cudaEventCreate(&ts4);

    // cudaEventRecord(ts0, NULL);

    wino_input_trans_chwn_suitFor128<<< dim3(blockn, blockn, chn), dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );  
    // cudaEventRecord(ts1, NULL);

    wino_kernel_trans_chwn_suitFor128<<< dim3(chn,1,1), dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);

    // change the sequence of two matrix of input and kernel.
    // GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, filterTran_gpu, inputTran_gpu,0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);
    
    wino_invers_chwn_suitFor128_2<<< dim3(numOfFilter,blockn,blockn), dim3(bat4Conv,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_chwn_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // cudaEventRecord(ts4, NULL);

    chn = numOfFilter;

    cudaDeviceSynchronize();
    // float t0, t1, t2, t3, total;
    // cudaEventElapsedTime(&t0, ts0, ts1);
    // cudaEventElapsedTime(&t1, ts1, ts2);
    // cudaEventElapsedTime(&t2, ts2, ts3);
    // cudaEventElapsedTime(&t3, ts3, ts4);
    // cudaEventDestroy(ts0);
    // cudaEventDestroy(ts1);
    // cudaEventDestroy(ts2);
    // cudaEventDestroy(ts3);
    // cudaEventDestroy(ts4);
    // total = t0+t1+t2+t3;

    // printf("time:%lf ms\t (%f, %f, %f, %f)\n", (total),(100*t0/total),(100*t1/total),(100*t2/total),(100*t3/total) );

}
*/

/* // DEBUG Code
__global__ void sumLayer(int bat4Conv,int chn, int inside, float *featureMap){
    float sum=0;
    for (int c=0;c<chn;c++){
        for (int i=0;i<inside;i++){
            for (int j=0;j<inside;j++){
                sum += featureMap[c*inside*inside+i*inside+j];
            }
        }
    }
    printf("the sum of the feature map:%f\n",sum);
}

__global__ void sumVar(int batch, int MN, int k, float *featureMap){
    float sum=0;
    for (int c=0;c<batch;c++){
        for (int i=0;i<MN;i++){
            for (int j=0;j<k;j++){
                sum += featureMap[c*MN*k+i*k+j];
            }
        }
    }
    printf("the sum of the tem Vars:%f\n",sum);
}
*/

void wrapedConv_CHWN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
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

    int nInputTran = 36 * MSize * KSize;
    int nFilterTran = 36 * NSize * KSize;
    int nGemmOutput = 36 * MSize*NSize;

    float *workSpace, *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    long long nWorkSpace = nInputTran + nFilterTran + nGemmOutput;

    cudaMalloc((void **) &workSpace,  nWorkSpace*sizeof(float));

    inputTran_gpu = workSpace;
    filterTran_gpu = workSpace + nInputTran;
    gemmOutput_gpu = workSpace + nInputTran + nFilterTran; 

    int blockx, blocky, bat4Gemm;

    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm =36;
    
    int oside = inside + 2*padding -2;

    // cudaEvent_t ts0, ts1, ts2, ts3, ts4;
    // cudaEventCreate(&ts0);
    // cudaEventCreate(&ts1);
    // cudaEventCreate(&ts2);
    // cudaEventCreate(&ts3);
    // cudaEventCreate(&ts4);

    // cudaEventRecord(ts0, NULL);

    // sumLayer<<<1,1>>>( bat4Conv, chn, inside, m1);
    // sumLayer<<<1,1>>>( bat4Conv, chn, inside, *output);

    wino_input_trans_chwn_suitFor128<<< dim3(blockn, blockn, chn), dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );  
    // cudaEventRecord(ts1, NULL);

    

    wino_kernel_trans_chwn_suitFor128<<< dim3(chn,1,1), dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);
    // cudaEventRecord(ts2, NULL);

    // cudaDeviceSynchronize();
    // sumVar<<<1,1>>>( 36, MSize, KSize, inputTran_gpu);
    // sumVar<<<1,1>>>( 36, NSize, KSize, filterTran_gpu);

    // change the sequence of two matrix of input and kernel.
    // GEMM_batch_256_128x128_KMKN<<< dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>>(MSize,NSize,KSize,1, inputTran_gpu, filterTran_gpu,0, gemmOutput_gpu);
    GEMM_batch_256_128x128_KMKN<<< dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>>(NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, bat4Gemm), dim3(256,1,1)>>> (NSize, MSize, KSize, 1, filterTran_gpu, inputTran_gpu, 0, gemmOutput_gpu);
    // cudaEventRecord(ts3, NULL);

    // cudaDeviceSynchronize();
    // sumVar<<<1,1>>>( 36, MSize, NSize, gemmOutput_gpu);
    
    wino_invers_chwn_suitFor128_2<<< dim3(numOfFilter,blockn,blockn), dim3(bat4Conv,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // wino_invers_chwn_suitFor128<<< dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // cudaEventRecord(ts4, NULL);
    
    // cudaDeviceSynchronize();
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu1 =(float *)malloc(nConvOutput * sizeof(float));
    // cudaMemcpy(convOutput_gpu1, *output, nConvOutput<<2, cudaMemcpyDeviceToHost);
    // int cnt4 = save_parameter("result/conv_CHWN.bin", nConvOutput, convOutput_gpu1);
    // free(convOutput_gpu1);

    chn = numOfFilter;

    

    cudaDeviceSynchronize();

    // float t0, t1, t2, t3, total;
    // cudaEventElapsedTime(&t0, ts0, ts1);
    // cudaEventElapsedTime(&t1, ts1, ts2);
    // cudaEventElapsedTime(&t2, ts2, ts3);
    // cudaEventElapsedTime(&t3, ts3, ts4);
    // cudaEventDestroy(ts0);
    // cudaEventDestroy(ts1);
    // cudaEventDestroy(ts2);
    // cudaEventDestroy(ts3);
    // cudaEventDestroy(ts4);
    // total = t0+t1+t2+t3;

    // printf("time:%lf ms\t (%f, %f, %f, %f)\n", (total),(100*t0/total),(100*t1/total),(100*t2/total),(100*t3/total) );
    cudaFree(workSpace);

}


/*
    In this design, one thread coresponding to a tile, and the threads inside a block corresponding to each bat4Conv of a channel
    The block config should be (blockx, blocky, channel).
    The M-dim of the GEMM should be arranged as blockx * blocky * bat4Conv, bat4Conv is the last dimension.
*/
__global__ void wino_input_trans_chwn_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x; // batch
    int bidx = blockIdx.x;  // blockn.x
    int bidy = blockIdx.y;  // blockn.y
    int bidz = blockIdx.z;  // chn_in

    int totalChn = gridDim.z;
    int numOfBatch = blockDim.x;
    int blockn = gridDim.x;

    float Mread[6][6] = {{0}};

    int inside = side;

    // take care of x-direction and y direction.
    pInputs = &pInputs[tidx + bidx*4*numOfBatch +bidy*4*inside*numOfBatch + bidz*inside*inside*numOfBatch - padding*(padding+inside)*numOfBatch ];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidx*numOfBatch + bidy*numOfBatch*blockn + bidz*MSize ];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidx + j >= padding)&&( 4*bidy +i >= padding)&&(4*bidx + j <=inside-1+padding) && (4*bidy +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j*numOfBatch + i * inside * numOfBatch];
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

__global__ void wino_kernel_trans_chwn_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // numOfFilter
    int bid  = blockIdx.x;   // chn_in

    int chn_out = blockDim.x;
    // int numOfFilter = blockDim.x;


    float Mread[3][3] ={{0}};
    pInputs = &pInputs[tidx + bid *3*3*chn_out ];
    pOutputs = &pOutputs[tidx + bid*NSize];

    #pragma unroll
    for (int i= 0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Mread[i][j] = pInputs[j*chn_out + i*3 *chn_out];
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

__global__ void wino_invers_chwn_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int chn = threadIdx.x;  // numOfFilter or chn_out
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
    pOutputs = &pOutputs[bidx + bidy*4*bat4Conv + bidz*4*oside*bat4Conv + chn*oside*oside*bat4Conv];

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
                pOutputs[i*oside*bat4Conv + j*bat4Conv] = Mread[i][j];
            }
        }
    }
}


__global__ void wino_invers_chwn_suitFor128_2(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int tidx = threadIdx.x;  // bat4Conv
    int bidx = blockIdx.x; // numOfFilter or chn_out
    int bidy = blockIdx.y; // blockn.x
    int bidz = blockIdx.z; // blockn,y

    float Mread[6][6] = {{0}};
    //take care, this should be the total MSize and NSize, rather than M and N;
    int size = MSize * NSize;
    int bat4Conv = blockDim.x;
    int chn_out = gridDim.x;
    int blockn = gridDim.y;

    pInputs = &pInputs[tidx + bidx*MSize + bidy* bat4Conv + bidz* bat4Conv*blockn ];
    // pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];
    pOutputs = &pOutputs[tidx + bidy*4*bat4Conv + bidz*4*oside*bat4Conv + bidx*oside*oside*bat4Conv];

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
                pOutputs[i*oside*bat4Conv + j*bat4Conv] = Mread[i][j];
            }
        }
    }
}