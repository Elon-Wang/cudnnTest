#include "GEMM.cuh"

__global__ void warmup(){}
void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding , float *m1, float *m2, float **output);
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs );
__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);
__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);

void wrapedConv_NCHW(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){
    
    // int padding =1;
    int size = numOfFilter * chn * (inside-2+2*padding) * (inside-2+2*padding) ;
    if (*output != NULL) {
        cudaFree(*output);
    }
    cudaMalloc(output, size *sizeof(float));
    assert(inside >=4);

    int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);

    int blockn = (inside_beta -2) /4;
    
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;
    // int K = chn;

    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    //So strange here, why all of this work when I add this "+1", need to figure out !!!
    // int nInput = bat4Conv * inside * inside * (chn) +1;
    // int nFilter = 9 * numOfFilter * chn;
    int nInputTran = 36 * MSize * KSize;
    int nFilterTran = 36 * NSize * KSize;

    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);

    float *inputTran_gpu, *filterTran_gpu;
    cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    
    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);

    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );

    // float *output1 =(float *)malloc(nInputTran * sizeof(float));
    // cudaMemcpy(output1, inputTran_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    // const char module1Name[] = "../data/output1.bin";
    // int cnt4 = save_parameter(module1Name, nInputTran, output1);

    wino_kernel_trans_nchw_suitFor128<<<dim3(chn,1,1),dim3(numOfFilter,1,1)>>>(NSize, KSize, m2, filterTran_gpu);

    //TBC, gemm part
    int blocky, blockx, bat4Gemm;
    float *gemmOutput_gpu;
    int nGemmOutput;
    blockx = (M+127)/128;
    blocky = (N+127)/128;
    bat4Gemm = 36;
    nGemmOutput = 36 * MSize*NSize;
    cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);
    GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1,inputTran_gpu,filterTran_gpu,0,gemmOutput_gpu);
    
    int oside = inside+2*padding - 2;
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);
    // wino_invers_chwn_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);
    wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, *output);
    // printf("blockn:%d\n",blockn);

    chn = numOfFilter;

    // cudaMemcpy(output, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToDevice);
    // __syncthreads();
    cudaFree(inputTran_gpu);
    cudaFree(filterTran_gpu);
    cudaFree(gemmOutput_gpu);
    // cudaFree(convOutput_gpu);
}


//OLD
// one thread corresponding to one layer
// and one block corresponding to one channel of all layer.

//NEW
// 1024 thread limit, hard to arrange the thread 
// one thread corresponding to one tile
// and one block corresponding to one tile of all batch.
// Problem: the thread acess is not continuely, and will cause serious delay problem, makes the problem super slow.
// wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;     // batch
    int bidx = blockIdx.x;     //  chn_in
    int bidy = blockIdx.y;     // blockn.x
    int bidz = blockIdx.z;     // blockn.y

    // if (tidx ==0 && tidy==0 && tidz==0 && bid ==0)
    //     printf("(bat4Conv,blockn,blockn,chn_in)=(%d,%d,%d,%d)\n",blockDim.z,blockDim.x,blockDim.y,gridDim.x);
    // printf("input_gpu[0]:%lf\n",pInputs[0]);

    int totalChn = gridDim.x;
    int numOfBatch = blockDim.x;
    int blockn = gridDim.y;

    float Mread[6][6] = {{0}};

    int inside = side;

    //take care of x-direction and y direction.
    pInputs = &pInputs[bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidy*numOfBatch + bidz *numOfBatch *blockn + bidx *MSize];
    
    
    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j + i * inside];
            }
        }
    }

    // DEBUG
    // if (tidx ==0 && bidx==0 && bidz==0 && bidy ==5){
    //     for( int i =0;i<6;i++) {
    //         for (int j=0; j<6;j++) {
    //             // if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
    //                 // printf("%f  ",Mread [i][j]);
    //                 printf("(%d,%d,%d,address:%d)  ",(4*bidy+j),(4*bidz+i),(4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding),(bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)+j + i * inside));
    //         }printf("\n");
    //     }
    // }

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

// WTF are you doing, this time is Apr 16, you didn't code even one line in the past months.
// you're such a fucking idiot.

// Modify at March 13, 2023
// try to use more thread rather using many blocks but only a few threads.

// wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, m1, inputTran_gpu );
wino_input_trans_nchw_suitFor128_new1<<<dim3(),dim3(8,8,chn), (inside*inside*8*4) >>>();
// chn cannot be larger than 16 then.
__global__ void wino_input_trans_nchw_suitFor128_new1(){
    int tidx = threadIdx.x;  // 0-8
    int tidy = threadIdx.y;  // 0-8
    int bidx = blockIdx.x;

    extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};

    pInputs = &pInputs[];

    for(){
        // store the data into the shared cache
        // constrain the range of the input
        inCache = (pInputs + );
    }

    // load the data from the shared into the private
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            Mread[i][j] = inCache[];
        }
    }

    // do calculation of each thread;
    // each thread calculate for 
    // each block claculate for
    float Atd[6][6] = {{0}};
    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }


    // store the result back to the output matrix;
    // int size = MSize * KSize;

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

wino_input_trans_nchw_suitFor128_new2<<<dim3(),dim3(blockn,blockn), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new2(){
    int tidx = threadIdx.x;  // 0-8
    int tidy = threadIdx.y;  // 0-8
    int bidx = blockIdx.x;

    extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};


}

// TODO: need to think about the computation efficiency. 
// And the access consistency of banks of the loading data.
// latency hidden as well. Does it need a shift of the input?
// wino_input_trans_nchw_suitFor128_new3<<<dim3(bat4Conv, blockn, blockn),dim3(chn,1,1), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tid = threadIdx.x;  // 0-1024, chn_in
    int bidx = blockIdx.x;  // batch
    int bidy = blockIdx.y;  // blockn.x
    int bidz = blockIdx.z;  // blockn.y

    // if (tidx ==0 && tidy==0 && tidz==0 && bid ==0)
    //     printf("(bat4Conv,blockn,blockn,chn_in)=(%d,%d,%d,%d)\n",blockDim.z,blockDim.x,blockDim.y,gridDim.x);
    // printf("input_gpu[0]:%lf\n",pInputs[0]);

    int totalChn = blockDim.x;
    int numOfBatch = gridDim.x;
    int blockn = gridDim.y;

    // extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};

    pInputs = &pInputs[bidy*4 + bidz*4*inside + tid*inside*inside + bidx*totalChn*inside*inside - padding*(1+inside) ];
    pOutputs = &pOutputs[bidx + bidy*numOfBatch + bidz*numOfBatch*blockn + tid*MSize];

    for(int i=0;i<6;i++) 
    {
        for(int j=0; j<6; j++)
        {
            if( ( 4*bidy+j<=padding )&&( 4*bidz+i>= padding )&&( 4*bidy+j<=inside-1+padding)&&(4*bidz+i<=inside-1+padding) )
            {
                Mread[i][j] = pInputs[j+i*inside];
            }
        }
    }

    float Atd[6][6] ={{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = MSize *KSize;

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

// wino_input_trans_nchw_suitFor128_new4<<<dim3(bat4Conv, blockn*blockn/(64), chn/4),dim3(8,8,4), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new4(){
    int tidx = threadIdx.x;  // 0-8,
    int tidy = threadIdx.y;  // 0-8
    int tidz = threadIdx.z;  // 0-4
    int bidx = blockIdx.x;  // batch
    int bidy = blockIdx.y;  // blockn by 64
    int bidz = blockIdx.z;  // chn by 4


    int totalChn = arg_Chn;
    int numofBatch = gridDim.x;
    int blockn = arg_blockn;

    float Mread[6][6] = {{0}};

    int inside = side;

    blocknX = bidy / (blockn/8);
    blocknY = bidy % (blockn/8);

    pInputs = &pInputs[tidx*4 +tidy*4*inside + tidz*inside*inside + bidx*inside*inside*totalChn+ blocknX*32 +blocknY*32*inside + bidz*inside*inside*4];

    pOutputs= &pOutputs[bidx + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*numOfBatch*blockn + blocknY*numOfBatch*blockn*8 + tidz*MSize + bidz*MSize*4];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j + i * inside];
            }
        }
    }

    // DEBUG
    // if (tidx ==0 && bidx==0 && bidz==0 && bidy ==5){
    //     for( int i =0;i<6;i++) {
    //         for (int j=0; j<6;j++) {
    //             // if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
    //                 // printf("%f  ",Mread [i][j]);
    //                 printf("(%d,%d,%d,address:%d)  ",(4*bidy+j),(4*bidz+i),(4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding),(bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)+j + i * inside));
    //         }printf("\n");
    //     }
    // }

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

// change the order of batch and the chn.
// wino_input_trans_nchw_suitFor128_new5<<<dim3(batch, blockn, blockn),dim3(chn,1,1), (inside*inside*8*4) >>>();
__global__ void wino_input_trans_nchw_suitFor128_new5(){
    int tidx = threadIdx.x;    // chn_in
    int bidx = blockIdx.x;     // batch
    int bidy = blockIdx.y;     // blockn.x
    int bidz = blockIdx.z;     // blockn.y

    // if (tidx ==0 && tidy==0 && tidz==0 && bid ==0)
    //     printf("(bat4Conv,blockn,blockn,chn_in)=(%d,%d,%d,%d)\n",blockDim.z,blockDim.x,blockDim.y,gridDim.x);
    // printf("input_gpu[0]:%lf\n",pInputs[0]);

    int totalChn = blockDim.x;
    int numOfBatch = gridDim.x;
    int blockn = gridDim.y;

    float Mread[6][6] = {{0}};

    int inside = side;

    //take care of x-direction and y direction.
    pInputs = &pInputs[bidy*4 + bidz*4*inside + tidx* inside *inside + bidx* totalChn*inside*inside - padding *(1+inside)];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[bidx + bidy*numOfBatch + bidz *numOfBatch *blockn + tidx *MSize];
    
    
    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
                Mread [i][j] = pInputs[j + i * inside];
            }
        }
    }

    // DEBUG
    // if (tidx ==0 && bidx==0 && bidz==0 && bidy ==5){
    //     for( int i =0;i<6;i++) {
    //         for (int j=0; j<6;j++) {
    //             // if ((4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding)) {
    //                 // printf("%f  ",Mread [i][j]);
    //                 printf("(%d,%d,%d,address:%d)  ",(4*bidy+j),(4*bidz+i),(4*bidy + j >= padding)&&( 4*bidz +i >= padding)&&(4*bidy + j <=inside-1+padding) && (4*bidz +i <= inside-1 +padding),(bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside - padding *(1+inside)+j + i * inside));
    //         }printf("\n");
    //     }
    // }

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

__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // numOfFilter
    int bid  = blockIdx.x;   // chn_in

    // int chn_out = blockDim.x;


    float Mread[3][3] ={{0}};
    pInputs = &pInputs[tidx*3*3*gridDim.x + bid *3*3];
    pOutputs = &pOutputs[tidx + bid*NSize];

    #pragma unroll
    for (int i= 0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Mread[i][j] = pInputs[j + i*3];
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

__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
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
    pOutputs = &pOutputs[chn* oside*oside + bidx*oside*oside*chn_out +bidy *4 + bidz*4*oside ];

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
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }
}