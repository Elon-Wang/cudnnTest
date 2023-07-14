#include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);

int main(int argc, char** argv){
    const char inputname[] = "../data/input.bin";
    const char filtername[] = "../data/filter.bin";

    int bat4Conv = atoi(argv[1]);
    int inside = atoi(argv[2]);
    int chn = atoi(argv[3]);
    int numOfFilter = atoi(argv[4]);
    int padding = 1;

    int nInput = bat4Conv * inside * inside * (chn) +1;

    float *input_cpu = get_parameter(inputname, nInput);
    float *input_gpu;  //, *output_gpu;
    cudaMalloc((void **) &input_gpu, nInput<<2);
    cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);

    // Output part
    // int oside = inside - 2 + 2 * padding;
    // int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    // float *convOutput_gpu;
    // cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    // printf("inside: %d, oside:%d\n",inside,oside);

    // wrapedConv_NCHW(bat4Conv, inside, chn, numOfFilter, padding , input_gpu, filter_gpu,convOutput_gpu);
    int marginOfInputSide = (inside+2*padding-6)%4;
    bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
    int inside_beta = sideCheck? inside +2*padding : (inside + 2*padding + 4 - marginOfInputSide);

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

    // Could we exempt this part? just using the parameter?
    // float *input_gpu, *filter_gpu;
    // cudaMalloc((void **) &input_gpu, nInput<<2);
    // cudaMalloc((void **) &filter_gpu, nFilter<<2);
    // cudaMemcpy(input_gpu, m1, nInput<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(filter_gpu, m2, nFilter<<2, cudaMemcpyDeviceToDevice);
    // The above part.

    // printf("inside_beta:%d  blockn:%d  M:%d MSize:%d nInputTran:%d\n",inside_beta,blockn, M,  MSize, nInputTran);

    float *inputTran1_gpu, *inputTran2_gpu;
    cudaMalloc((void **) &inputTran1_gpu,  nInputTran<<2);
    cudaMalloc((void **) &inputTran2_gpu,  nInputTran<<2);
    
    // wino_input_trans_chwn_suitFor128<<<dim3(blockn,blockn,chn),dim3(bat4Conv,1,1) >>> (inside, inside_beta, MSize, KSize, m1, inputTran_gpu );
    // wino_kernel_trans_chwn_suitFor128<<<dim3(chn,1,1) , dim3(numOfFilter,1,1) >>>( NSize, KSize, m2, filterTran_gpu);

    wino_input_trans_nchw_suitFor128<<<dim3(chn,blockn,blockn),dim3(bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, input_gpu, inputTran1_gpu );
    wino_input_trans_nchw_suitFor128_new3<<<dim3(bat4Conv,blockn,blockn),dim3(chn,1,1)>>>(inside, inside_beta, MSize, KSize, padding, input_gpu, inputTran2_gpu );

    float *inputTran1_cpu = (float *)malloc(nInputTran * sizeof(float));
    cudaMemcpy(inputTran1_cpu, inputTran1_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);
    float *inputTran2_cpu = (float *)malloc(nInputTran * sizeof(float));
    cudaMemcpy(inputTran2_cpu, inputTran2_gpu, nInputTran<<2, cudaMemcpyDeviceToHost);

    printf("inputTran1_cpu[0]:%lf\n",inputTran1_cpu[0]);
    printf("inputTran2_cpu[0]:%lf\n",inputTran2_cpu[0]);
    if ( inputTran2_cpu[0] == 0.0f) {
        //printf("Test3:\n 1:%f\t 2:%f\n",output_cpu[0], output_cpu[1]);
        printf("moduleConv error\n");
    }

    //used for debug;
    const char finalName1[] = "../data/CuOutput1.bin";
    const char finalName2[] = "../data/CuOutput2.bin";
    int cnt4 = save_parameter(finalName1, nInputTran, inputTran1_cpu);
    int cnt5 = save_parameter(finalName2, nInputTran, inputTran2_cpu);

    return 0;
}

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

    int inside = side;

    pInputs = &pInputs[bidy*4 + bidz*4*inside + tid*inside*inside + bidx*totalChn*inside*inside - padding*(1+inside) ];
    pOutputs = &pOutputs[bidx + bidy*numOfBatch + bidz*numOfBatch*blockn + tid*MSize];

    for(int i=0;i<6;i++) 
    {
        for(int j=0; j<6; j++)
        {
            if( ( 4*bidy+j>=padding )&&( 4*bidz+i>= padding )&&( 4*bidy+j<=inside-1+padding)&&(4*bidz+i<=inside-1+padding) )
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

__global__ void wino_input_trans_nchw_suitFor128_new4(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs, int arg_blockn, int arg_Chn){
    int tidx = threadIdx.x;  // 0-8,
    int tidy = threadIdx.y;  // 0-8
    int tidz = threadIdx.z;  // 0-4
    int bidx = blockIdx.x;   // batch
    int bidy = blockIdx.y;   // blockn by 64
    int bidz = blockIdx.z;   // chn by 4


    int totalChn = arg_Chn;
    int numofBatch = gridDim.x;
    int blockn = arg_blockn;

    float Mread[6][6] = {{0}};

    int inside = side;

    blocknX = bidy % (blockn/8);
    blocknY = bidy / (blockn/8);

    pInputs = &pInputs[tidx*4 +tidy*4*inside + tidz*inside*inside + bidx*inside*inside*totalChn+ blocknX*32 +blocknY*32*inside + bidz*inside*inside*4];

    pOutputs= &pOutputs[bidx + tidx*numOfBatch + blocknX*8*numOfBatch + tidy*numOfBatch*blockn + blocknY*numOfBatch*blockn*8 + tidz*MSize + bidz*MSize*4];

    // Try float4 or float3, how to compatible float4 with flexible inside
    for( int i =0;i<6;i++) {
        for (int j=0; j<6;j++) {
            if ((4*tidx + j + blocknX*32 >= padding)&&( 4*tidy +i + blocknY*32 >= padding)&&(4*bidy + j + blocknX*32 <=inside-1+padding) && (4*bidz + i + blocknY*32 <= inside-1 +padding)) {
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

__global__ void wino_input_trans_nchw_suitFor128_new1(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs){
    int tidx = threadIdx.x;  // 0-8
    int tidy = threadIdx.y;  // 0-8
    int bidx = blockIdx.x;  // chn
    int bidy = blockIdx.y;  // bat4Conv

    int blockn = 1+(inside -6)/4;

    int totalChn = gridDim.x;

    extern __shared__ float inCache[];
    float Mread[6][6] = {{0}};

    pInputs = &pInputs[bidx*inside*inside + bidy*totalChn *inside*inside + 4*tidx + ];

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