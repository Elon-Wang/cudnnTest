#define Bccum(sum,a,num1,b,num2,c)    \
{    \
    sum[4*a].x = fma(num1[b],num2[c],sum[4*a].x);\
    sum[4*a].y = fma(num1[b],num2[c+1],sum[4*a].y);\
    sum[4*a].z = fma(num1[b],num2[c+2],sum[4*a].z);\
    sum[4*a].w = fma(num1[b],num2[c+3],sum[4*a].w);\
    \
    sum[4*a+1].x = fma(num1[b+1],num2[c],sum[4*a+1].x);\
    sum[4*a+1].y = fma(num1[b+1],num2[c+1],sum[4*a+1].y);\
    sum[4*a+1].z = fma(num1[b+1],num2[c+2],sum[4*a+1].z);\
    sum[4*a+1].w = fma(num1[b+1],num2[c+3],sum[4*a+1].w);\
    \
    sum[4*a+2].x = fma(num1[b+2],num2[c],sum[4*a+2].x);\
    sum[4*a+2].y = fma(num1[b+2],num2[c+1],sum[4*a+2].y);\
    sum[4*a+2].z = fma(num1[b+2],num2[c+2],sum[4*a+2].z);\
    sum[4*a+2].w = fma(num1[b+2],num2[c+3],sum[4*a+2].w);\
    \
    sum[4*a+3].x = fma(num1[b+3],num2[c],sum[4*a+3].x);\
    sum[4*a+3].y = fma(num1[b+3],num2[c+1],sum[4*a+3].y);\
    sum[4*a+3].z = fma(num1[b+3],num2[c+2],sum[4*a+3].z);\
    sum[4*a+3].w = fma(num1[b+3],num2[c+3],sum[4*a+3].w);\
}

#define Tranload(Cache,a,b,reg)    \
{    \
    Cache[b*4][a] = reg.x;\
    Cache[b*4+1][a] = reg.y;\
    Cache[b*4+2][a] = reg.z;\
    Cache[b*4+3][a] = reg.w;\
}

#define Accum(sum,a,num1,b,num2,c)    \
{    \
    sum[4*a].x = fma(num1[b],num2[c].x,sum[4*a].x);\
    sum[4*a].y = fma(num1[b],num2[c].y,sum[4*a].y);\
    sum[4*a].z = fma(num1[b],num2[c].z,sum[4*a].z);\
    sum[4*a].w = fma(num1[b],num2[c].w,sum[4*a].w);\
    \
    sum[4*a+1].x = fma(num1[b+1],num2[c].x,sum[4*a+1].x);\
    sum[4*a+1].y = fma(num1[b+1],num2[c].y,sum[4*a+1].y);\
    sum[4*a+1].z = fma(num1[b+1],num2[c].z,sum[4*a+1].z);\
    sum[4*a+1].w = fma(num1[b+1],num2[c].w,sum[4*a+1].w);\
    \
    sum[4*a+2].x = fma(num1[b+2],num2[c].x,sum[4*a+2].x);\
    sum[4*a+2].y = fma(num1[b+2],num2[c].y,sum[4*a+2].y);\
    sum[4*a+2].z = fma(num1[b+2],num2[c].z,sum[4*a+2].z);\
    sum[4*a+2].w = fma(num1[b+2],num2[c].w,sum[4*a+2].w);\
    \
    sum[4*a+3].x = fma(num1[b+3],num2[c].x,sum[4*a+3].x);\
    sum[4*a+3].y = fma(num1[b+3],num2[c].y,sum[4*a+3].y);\
    sum[4*a+3].z = fma(num1[b+3],num2[c].z,sum[4*a+3].z);\
    sum[4*a+3].w = fma(num1[b+3],num2[c].w,sum[4*a+3].w);\
}

#define Store4x4(reg,b,global,c,k) \
{\
  *((float4 *)(global+c)) = reg[4*b]; \
  *((float4 *)(global+c+k)) = reg[4*b+1]; \
  *((float4 *)(global+c+2*k)) = reg[4*b+2]; \
  *((float4 *)(global+c+3*k)) = reg[4*b+3]; \
}

#define muladd4x4(alpha,reg,beta,reg1,b) \
{\
  reg[4*b].x = alpha*reg[4*b].x + beta*reg1[4*b].x; \
  reg[4*b].y = alpha*reg[4*b].y + beta*reg1[4*b].y; \
  reg[4*b].z = alpha*reg[4*b].z + beta*reg1[4*b].z; \
  reg[4*b].w = alpha*reg[4*b].w + beta*reg1[4*b].w; \
    \
  reg[4*b+1].x = alpha*reg[4*b+1].x + beta*reg1[4*b+1].x; \
  reg[4*b+1].y = alpha*reg[4*b+1].y + beta*reg1[4*b+1].y; \
  reg[4*b+1].z = alpha*reg[4*b+1].z + beta*reg1[4*b+1].z; \
  reg[4*b+1].w = alpha*reg[4*b+1].w + beta*reg1[4*b+1].w; \
  \
  reg[4*b+2].x = alpha*reg[4*b+2].x + beta*reg1[4*b+2].x; \
  reg[4*b+2].y = alpha*reg[4*b+2].y + beta*reg1[4*b+2].y; \
  reg[4*b+2].z = alpha*reg[4*b+2].z + beta*reg1[4*b+2].z; \
  reg[4*b+2].w = alpha*reg[4*b+2].w + beta*reg1[4*b+2].w; \
  \
  reg[4*b+3].x = alpha*reg[4*b+3].x + beta*reg1[4*b+3].x; \
  reg[4*b+3].y = alpha*reg[4*b+3].y + beta*reg1[4*b+3].y; \
  reg[4*b+3].z = alpha*reg[4*b+3].z + beta*reg1[4*b+3].z; \
  reg[4*b+3].w = alpha*reg[4*b+3].w + beta*reg1[4*b+3].w; \
  \
}

#define Load4x4(global,c,k,reg,b) \
{\
  reg[4*b] = *((float4 *)(global+c)); \
  reg[4*b+1] = *((float4 *)(global+c+k)); \
  reg[4*b+2] = *((float4 *)(global+c+2*k)); \
  reg[4*b+3] = *((float4 *)(global+c+3*k)); \
}

int gcd(int a, int b){
 return a % b ? gcd(b, a % b) : b;
}

int lcm(int a, int b){
 return a * b / gcd(a, b);
}


__global__ void warmup(){}

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

    // if (bid==0 && tidx ==  16) {
    //     printf("input pos:%d\n",tidx*3*3*gridDim.x + bid *3*3);
    //     printf("val:%f\n",pInputs[tidx*3*3*gridDim.x + bid *3*3]);
    // }

    for(int i=0;i<6;i++){
        pOutputs[i*6*size] = Gg[i][0]/4;
        pOutputs[i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[i*6*size + 5*size] = Gg[i][2];
    }
}

__global__ void GEMM_batch_256_128x128_KMKN(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    int K = chn;

    int mx = blockIdx.x;
    int my = blockIdx.y;
    int Batch = blockIdx.z;//128
    
    int tid = threadIdx.x;
    int tx = tid%16;//16
    int ty = tid/16;//16


    int xload = tid%32;
    int yload = tid/32;

    // int a = tx*4 + ty*4*N;

    m1= &m1[Batch*M*K + mx*128];
    m2= &m2[Batch*N*K + my*128];
    output= &output[Batch*M*N + mx*128*N + my*128 + tx*4 + ty*4*N];

    __shared__ float m1Cache[2][8][128];
    __shared__ float m2Cache[2][8][128];
    
    float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    float in1[2][8] = {{0}};
    float in2[2][8] = {{0}};

    float4 reg_m1;
    float4 reg_m2;

    //load first tile to m1 and m2
    float4 * m1_load =  (float4 *)(m1 + (xload)*4 + (yload)*M);
    float4 * m2_load =  (float4 *)(m2 + (xload)*4 + (yload)*N);

    // load to shared
    reg_m1 = *(m1_load);
    reg_m2 = *(m2_load);
    *((float4 *)(m1Cache[0][yload] + xload*4)) = reg_m1;
    *((float4 *)(m2Cache[0][yload] + xload*4)) = reg_m2;

    __syncthreads();

    //load first data to reg
    *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
    *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 64));

    *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
    *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 64));

    int sm = 0;
    int write_idx = 1;

    do{
        sm += 8;
        // m1_load += 2;
        // m2_load += 2;
        if(sm < K){
            //load next tile from global to reg
            // m1_load += 2;
            // m2_load += 2;
            reg_m1 = *(m1_load + sm*M/4);
            reg_m2 = *(m2_load + sm*N/4);
        }
        //load shared to reg
        int load_idx = write_idx ^ 1; 
        int ind = 1;

        #pragma unroll
        for(int x=1;x<8;++x){
            *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
            *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 64));

            *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
            *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 64));

            Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
            Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
            Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
            Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);

            ind^=1;
        }

        //load next tile to shared
        *((float4 *)(m1Cache[write_idx][yload] + xload*4)) = reg_m1;
        *((float4 *)(m2Cache[write_idx][yload] + xload*4)) = reg_m2;
        __syncthreads();
    
        write_idx ^= 1;

        //load shared to reg
        *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
        *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 64));

        *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
        *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 64));

        // muladd4x4(1,sum,0,Rsum,0);
        // muladd4x4(1,sum,0,Rsum,1);
        // muladd4x4(1,sum,0,Rsum,2);
        // muladd4x4(1,sum,0,Rsum,3);

        //last tile
        Bccum(sum,0,in1[1],0,in2[1],0);
        Bccum(sum,1,in1[1],0,in2[1],4);
        Bccum(sum,2,in1[1],4,in2[1],0);
        Bccum(sum,3,in1[1],4,in2[1],4);

    }while(sm<K);
    
    // //load C
    // Load4x4(output+a,0,N,Rsum,0);
    // Load4x4(output+a,64,N,Rsum,1);
    // Load4x4(output+a,64*N,N,Rsum,2);
    // Load4x4(output+a,64*N+64,N,Rsum,3);
    // cal
    // muladd4x4(1,sum,0,Rsum,0);
    // muladd4x4(1,sum,0,Rsum,1);
    // muladd4x4(1,sum,0,Rsum,2);
    // muladd4x4(1,sum,0,Rsum,3);
    //write back
    Store4x4(sum,0,output,0,N);
    output+=64;
    Store4x4(sum,1,output,0,N);
    output+=64*N-64;
    Store4x4(sum,2,output,0,N);
    output+=64;
    Store4x4(sum,3,output,0,N);
}

__global__ void wino_invers_nchw_suitFor128(int oside, int MSize, int NSize, float* pInputs, float* pOutputs) {
    int chn = threadIdx.x;  // numOfFilter or chn_out
    int bidx = blockIdx.x; // bat4Conv 
    int bidy = blockIdx.y; // blockn.x
    int bidz = blockIdx.z; // blockn.y


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

    // if (chn ==0 && bidx ==0 && bidy ==0 && bidz==0) {
    //     printf("oside:%d\n",oside);
    //     for (int i=0;i<4;i++) {
    //         for (int j=0; j<4; j++) {
    //             printf("%d ",i*oside + j);
    //         }
    //         printf("\n");
    //     }
    // }
    

    for (int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            if( ( 4*bidy+j<=oside-1 ) && ( 4*bidz+i<=oside-1 ) ){
                pOutputs[i*oside + j] = Mread[i][j];
            }
        }
    }
}