
#include "GEMM.cuh"
// #include "winotrans.cuh"

// function defination
__global__ void warmup(){}
void wrapedConv_NCHW(int bat4Conv, int inside, int chn, int numOfFilter, float *m1, float *m2, float *output);
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, float * pInputs, float* pOutputs );
__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs);
__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs);

int main(int argc, char** argv){
    const char inputname[] = "../data/input.bin";
    const char filtername[] = "../data/filter.bin";

    int bat4Conv = atoi(argv[1]);
    int inside = atoi(argv[2]);
    int chn = atoi(argv[3]);
    int numOfFilter = atoi(argv[4]);

    //So strange here, why all of this work when I add this "+1", need to figure out !!!
    int nInput = bat4Conv * inside * inside * (chn) +1;
    int nFilter = 9 * numOfFilter * chn;

    float *input_cpu = get_parameter(inputname, nInput);
    float *filter_cpu = get_parameter(filtername, nFilter);
    float *input_gpu, *filter_gpu;  //, *output_gpu;
    cudaMalloc((void **) &input_gpu, nInput<<2);
    cudaMalloc((void **) &filter_gpu, nFilter<<2);
    cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, filter_cpu, nFilter<<2, cudaMemcpyHostToDevice);

    // Output part
    int oside = inside - 2;
    int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    float *convOutput_gpu;
    cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);
    
    wrapedConv_NCHW(bat4Conv, inside, chn, numOfFilter, input_gpu, filter_gpu,convOutput_gpu);

    float *convOutput_cpu = (float *)malloc(nConvOutput * sizeof(float));
    cudaMemcpy(convOutput_cpu, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToHost);
    
    printf("convOutput_cpu[0]:%lf\n",convOutput_cpu[0]);
    if ( convOutput_cpu[0] == 0.0f) {
        //printf("Test3:\n 1:%f\t 2:%f\n",output_cpu[0], output_cpu[1]);
        printf("moduleConv error\n");
    }

    //used for debug;
    const char finalName[] = "../data/ConvModule_NCHW.bin";
    int cnt4 = save_parameter(finalName, nConvOutput, convOutput_cpu);

    return 0;
}

void wrapedConv_NCHW(int bat4Conv, int inside, int chn, int numOfFilter, float *m1, float *m2, float *output){
    int marginOfInputSide = (inside-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside: (inside + 4 - marginOfInputSide);

    int blockn = (inside_beta -2) /4;
    
    int M = bat4Conv * blockn * blockn;
    int N = numOfFilter;
    int K = chn;

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

    float *inputTran_gpu, *filterTran_gpu;
    cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
    cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
    
    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);

    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
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
    
    int oside = inside - 2;
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);
    // wino_invers_chwn_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, convOutput_gpu);
    wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, output);
    // printf("blockn:%d\n",blockn);

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
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, float * pInputs, float* pOutputs){
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
    pInputs = &pInputs[bidy*4 + bidz*4*inside + bidx* inside *inside + tidx* totalChn*inside*inside];
    // take care  of the M = blockn * blockn * batch, and the order of them.
    pOutputs = &pOutputs[tidx + bidy*numOfBatch + bidz *numOfBatch *blockn + bidx *MSize];
    

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*bidy + j <=inside-1) && (4*bidz +i <= inside-1)) {
                Mread [i][j] = pInputs[j + i * inside];
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

__global__ void wino_kernel_trans_nchw_suitFor128(int NSize, int KSize, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // numOfFilter
    int bid  = blockIdx.x;   // chn_in

    int chn_out = blockDim.x;


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