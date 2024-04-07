#include "util.h"

__global__ void GEMM_batch_256_128x128_MKNK(int m, int n, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    int K = chn;
    int M = m;
    int N = n;

    int mx = blockIdx.x;
    int my = blockIdx.y;
    int Batch = blockIdx.z;

    int tid = threadIdx.x;
    int tx = tid%16;
    int ty = tid/16;

    // TODO: considering the hebing access
    int xload1 = tid%128;
    int yload1 = tid/128;

    int xload2 = tid%128;
    int yload2 = tid/128;

    m1 = &m1[Batch*M*K + mx*128*K];
    m2 = &m2[Batch*N*K + my*128*K];
    output = &output[Batch*M*N + mx*128*N + my*128+ tx*4 + ty*4*N];

    __shared__ float m1Cache[2][8][128];
    __shared__ float m2Cache[2][8][128];

    float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    float in1[2][8] = {{0}};
    float in2[2][8] = {{0}};

    float4 reg_m1;
    float4 reg_m2;

    //load first tile to m1 and m2
    float4 * m1_load =  (float4 *)(m1 + (xload1)*K + (yload1)*4);
    float4 * m2_load =  (float4 *)(m2 + (xload2)*K + (yload2)*4);

    reg_m1 = *(m1_load);
    reg_m2 = *(m2_load);

    Tranload(m1Cache[0],xload1,yload1,reg_m1);
    Tranload(m2Cache[0],xload2,yload2,reg_m2);

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
        m1_load+=2;
        m2_load+=2;
        if(sm < K){
            //load next tile from global to reg
            reg_m1 = *(m1_load);
            reg_m2 = *(m2_load);
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
        //*((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
        Tranload(m2Cache[write_idx],xload2,yload2,reg_m2);
        Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);

        __syncthreads();
    
        write_idx ^= 1;

        //load shared to reg
        *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
        *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 64));

        *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
        *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 64));

        //last tile
        Bccum(sum,0,in1[1],0,in2[1],0);
        Bccum(sum,1,in1[1],0,in2[1],4);
        Bccum(sum,2,in1[1],4,in2[1],0);
        Bccum(sum,3,in1[1],4,in2[1],4);

    }while(sm<K);

    //write back
    Store4x4(sum,0,output,0,N);
    output+=64;
    Store4x4(sum,1,output,0,N);
    output+=64*N-64;
    Store4x4(sum,2,output,0,N);
    output+=64;
    Store4x4(sum,3,output,0,N);

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

__global__ void GEMM_batch_256_128x128_MKKN(int m, int n, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    int K = chn;
    int M = m;
    int N = n;

    int mx = blockIdx.x;
    int my = blockIdx.y;
    int Batch = blockIdx.z;//128
    
    int tid = threadIdx.x;
    int tx = tid%16;//16
    int ty = tid/16;//16

    int xload1 = tid%128;
    int yload1 = tid/128;

    int xload2 = tid%32;
    int yload2 = tid/32;

    // int a = tx*4 + ty*4*N;

    m1= &m1[Batch*M*K + mx*128*K];
    m2= &m2[Batch*N*K + my*128];
    output= &output[Batch*M*N + mx*128*N + my*128 + tx*4 + ty*4*N];

    __shared__ float m1Cache[2][8][128];
    __shared__ float m2Cache[2][8][128];
    
    float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    float in1[2][8] = {{0}};
    float in2[2][8] = {{0}};

    float4 reg_m1;
    float4 reg_m2;

    //load first tile to m1 and m2
    float4 * m1_load =  (float4 *)(m1 + (xload1)*K + (yload1)*4);
    float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

    // load to shared
    reg_m1 = *(m1_load);
    reg_m2 = *(m2_load);
    Tranload(m1Cache[0],xload1,yload1,reg_m1);
    *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;

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
        m1_load+=2;
        if(sm < K){
            //load next tile from global to reg
            reg_m1 = *(m1_load);
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
        *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
        Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);

        __syncthreads();
    
        write_idx ^= 1;

        //load shared to reg
        *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
        *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 64));

        *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
        *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 64));

        //last tile
        Bccum(sum,0,in1[1],0,in2[1],0);
        Bccum(sum,1,in1[1],0,in2[1],4);
        Bccum(sum,2,in1[1],4,in2[1],0);
        Bccum(sum,3,in1[1],4,in2[1],4);

    }while(sm<K);

    //write back
    Store4x4(sum,0,output,0,N);
    output+=64;
    Store4x4(sum,1,output,0,N);
    output+=64*N-64;
    Store4x4(sum,2,output,0,N);
    output+=64;
    Store4x4(sum,3,output,0,N);

}

__global__ void GEMM_batch_128_128x64_KMKN(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    int K = chn;
  
    int mx = blockIdx.x;
    int my = blockIdx.y;
    int Batch = blockIdx.z;
    
    int tid = threadIdx.x;
    int tx = tid%8;//8
    int ty = tid/8;//16
  
    int xload1 = tid%32;//32
    int yload1 = tid/32;//4
  
    int xload2 = tid%16;
    int yload2 = tid/16;
  
    // int a = tx*4 + ty*4*N;
  
    m1= &m1[Batch*M*K + mx*128];
    m2= &m2[Batch*N*K + my*64];
    output= &output[Batch*M*N + mx*128*N + my*64 + tx*4 + ty*4*N];
  
    __shared__ float m1Cache[2][8][128];
    __shared__ float m2Cache[2][8][64];
    
    float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    float in1[2][8] = {{0}};
    float in2[2][8] = {{0}};
  
    float4 reg_m1_1, reg_m1_2;
    float4 reg_m2;
  
    //load first tile to m1 and m2
    float4 * m1_load =  (float4 *)(m1 + (xload1)*4 + yload1*M);
    float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);
  
    // load to shared
    reg_m1_1 = *(m1_load);
    reg_m1_2 = *(m1_load+M);
    reg_m2 = *(m2_load);
  
    *((float4 *)(m1Cache[0][yload1] + xload1*4)) = reg_m1_1;
    *((float4 *)(m1Cache[0][yload1+4] + xload1*4)) = reg_m1_2;
    *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;
  
    __syncthreads();
  
    //load first data to reg
    *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
    *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 32));
  
    *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
    *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 64));
  
    int sm = 0;
    int write_idx = 1;
  
    do{
        sm += 8;
        // m1_load += 2;
        if(sm < K){
            //load next tile from global to reg
            reg_m1_1 = *(m1_load + sm*M/4);
            reg_m1_2 = *(m1_load + sm*M/4 + M);
            reg_m2 = *(m2_load + sm*N/4);
        }
        //load shared to reg
        int load_idx = write_idx ^ 1; 
        int ind = 1;
  
        #pragma unroll
        for(int x=1;x<8;++x){
            *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
            *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 32));
  
            *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
            *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 64));
  
            Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
            Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
            Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
            Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);
  
            ind^=1;
        }
  
        //load next tile to shared
        *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
        *((float4 *)(m1Cache[write_idx][yload1] + xload1*4)) = reg_m1_1;
        *((float4 *)(m1Cache[write_idx][yload1] + xload1*4)) = reg_m1_2;
  
        __syncthreads();
    
        write_idx ^= 1;
  
        //load shared to reg
        *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
        *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 32));
  
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
    output+=32;
    Store4x4(sum,1,output,0,N);
    output+=64*N-32;
    Store4x4(sum,2,output,0,N);
    output+=32;
    Store4x4(sum,3,output,0,N);
  
}

__global__ void GEMM_batch_128_128x64(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;
  int tx = tid%8;//8
  int ty = tid/8;//16

  int xload1 = tid;

  int xload2 = tid%16;
  int yload2 = tid/16;

  // int a = tx*4 + ty*4*N;

  m1= &m1[Batch*M*K + mx*128*K];
  m2= &m2[Batch*N*K + my*64];
  output= &output[Batch*M*N + mx*128*N + my*64 + tx*4 + ty*4*N];

  __shared__ float m1Cache[2][8][128];
  __shared__ float m2Cache[2][8][64];
  
  float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2][8] = {{0}};
  float in2[2][8] = {{0}};

  float4 reg_m1_1, reg_m1_2;
  float4 reg_m2;

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1_1 = *(m1_load);
  reg_m1_2 = *(m1_load+1);
  reg_m2 = *(m2_load);

  Tranload(m1Cache[0],xload1,0,reg_m1_1);
  Tranload(m1Cache[0],xload1,1,reg_m1_2);
  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 32));

  *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
  *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 64));

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 8;
      m1_load += 2;
      if(sm < K){
          //load next tile from global to reg
          reg_m1_1 = *(m1_load);
          reg_m1_2 = *(m1_load + 1);
          reg_m2 = *(m2_load + sm*N/4);
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<8;++x){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 32));

          *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
          *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 64));

          Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
          Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
          Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
          Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);

          ind^=1;
      }

      //load next tile to shared
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
      Tranload(m1Cache[write_idx],xload1,0,reg_m1_1);
      Tranload(m1Cache[write_idx],xload1,1,reg_m1_2);

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 32));

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
  output+=32;
  Store4x4(sum,1,output,0,N);
  output+=64*N-32;
  Store4x4(sum,2,output,0,N);
  output+=32;
  Store4x4(sum,3,output,0,N);

}

__global__ void GEMM_batch_128_64x128(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;
  int tx = tid%16;//16
  int ty = tid/16;//8

  int xload1 = tid%64;//64
  int yload1 = tid/64;//2

  int xload2 = tid%32;//32
  int yload2 = tid/32;//4

  // int a = tx*4 + ty*4*N;

  m1= &m1[Batch*M*K + mx*64*K];
  m2= &m2[Batch*N*K + my*128];
  output= &output[Batch*M*N + mx*64*N + my*128 + tx*4 + ty*4*N];

  __shared__ float m1Cache[2][8][64];
  __shared__ float m2Cache[2][8][128];
  
  float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2][8] = {{0}};
  float in2[2][8] = {{0}};

  float4 reg_m1;
  float4 reg_m2_1, reg_m2_2;

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K + (yload1)*4);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1 = *(m1_load);
  reg_m2_1 = *(m2_load);
  reg_m2_2 = *(m2_load+N);

  Tranload(m1Cache[0],xload1,yload1,reg_m1);
  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2_1;
  *((float4 *)(m2Cache[0][yload2+4] + xload2*4)) = reg_m2_2;

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 64));

  *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
  *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 32));

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 8;

      m1_load+=2;
      m2_load+=2*N;
      if(sm < K){
          //load next tile from global to reg
          reg_m1 = *(m1_load);
          reg_m2_1 = *(m2_load);
          reg_m2_2 = *(m2_load + N);
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<8;++x){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 64));

          *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
          *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 32));

          Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
          Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
          Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
          Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);

          ind^=1;
      }

      //load next tile to shared
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2_1;
      *((float4 *)(m2Cache[write_idx][yload2+4] + xload2*4)) = reg_m2_2;
      Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 64));

      *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
      *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 32));

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
  output+=32*N-64;
  Store4x4(sum,2,output,0,N);
  output+=64;
  Store4x4(sum,3,output,0,N);

}

__global__ void GEMM_batch_128_64x128_KMKN(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    int K = chn;
  
    int mx = blockIdx.x;
    int my = blockIdx.y;
    int Batch = blockIdx.z;
    
    int tid = threadIdx.x;
    int tx = tid%16;//16
    int ty = tid/16;//8
  
    int xload1 = tid%16;//64
    int yload1 = tid/16;//2
  
    int xload2 = tid%32;//32
    int yload2 = tid/32;//4
  
    // int a = tx*4 + ty*4*N;
  
    m1= &m1[Batch*M*K + mx*64];
    m2= &m2[Batch*N*K + my*128];
    output= &output[Batch*M*N + mx*64*N + my*128 + tx*4 + ty*4*N];
  
    __shared__ float m1Cache[2][8][64];
    __shared__ float m2Cache[2][8][128];
    
    float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
    float in1[2][8] = {{0}};
    float in2[2][8] = {{0}};
  
    float4 reg_m1;
    float4 reg_m2_1, reg_m2_2;
  
    //load first tile to m1 and m2
    float4 * m1_load =  (float4 *)(m1 + (xload1)*4 + (yload1)*M);
    float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);
  
    // load to shared
    reg_m1 = *(m1_load);
    reg_m2_1 = *(m2_load);
    reg_m2_2 = *(m2_load+N);
  
    // Tranload(m1Cache[0],xload1,yload1,reg_m1);
    *((float4 *)(m1Cache[0][yload1] + xload1*4)) = reg_m1;
    *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2_1;
    *((float4 *)(m2Cache[0][yload2+4] + xload2*4)) = reg_m2_2;
  
    __syncthreads();
  
    //load first data to reg
    *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
    *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 64));
  
    *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
    *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 32));
  
    int sm = 0;
    int write_idx = 1;
  
    do{
        sm += 8;
        m1_load+=2*M;
        m2_load+=2*N;
        if(sm < K){
            //load next tile from global to reg
            // m1_load+=2*M;
            // m2_load+=2*N;
            reg_m1 = *(m1_load);
            reg_m2_1 = *(m2_load);
            reg_m2_2 = *(m2_load + N);
        }
        //load shared to reg
        int load_idx = write_idx ^ 1; 
        int ind = 1;
  
        #pragma unroll
        for(int x=1;x<8;++x){
            *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
            *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 64));
  
            *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
            *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 32));
  
            Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
            Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
            Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
            Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);
  
            ind^=1;
        }
  
        //load next tile to shared
        *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2_1;
        *((float4 *)(m2Cache[write_idx][yload2+4] + xload2*4)) = reg_m2_2;
        *((float4 *)(m1Cache[write_idx][yload1] + xload1*4)) = reg_m1;
        // Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);
  
        __syncthreads();
    
        write_idx ^= 1;
  
        //load shared to reg
        *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
        *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 64));
  
        *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
        *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 32));
  
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
    output+=32*N-64;
    Store4x4(sum,2,output,0,N);
    output+=64;
    Store4x4(sum,3,output,0,N);
  
}

__global__ void GEMM_batch_64_128x32(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;//64
  int tx = tid%4;//4
  int ty = tid/4;//16

  int xload1 = tid;
  // int yload1 = tid/128;

  int xload2 = tid%8;//8
  int yload2 = tid/8;//8


  m1= &m1[Batch*M*K + mx*128*K];
  m2= &m2[Batch*N*K + my*32];
  output= &output[Batch*M*N + mx*128*N + my*32 + tx*4 + ty*4*N];

  __shared__ float m1Cache[2][8][128];
  __shared__ float m2Cache[2][8][32];
  
  float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2][8] = {{0}};
  float in2[2][8] = {{0}};

  float4 reg_m1[4];
  float4 reg_m2;

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1[0] = *(m1_load);
  reg_m1[1] = *(m1_load+1);
  reg_m1[2] = *(m1_load+16*K);
  reg_m1[3] = *(m1_load+16*K+1);

  reg_m2 = *(m2_load);

  Tranload(m1Cache[0],xload1,0,reg_m1[0]);
  Tranload(m1Cache[0],xload1,1,reg_m1[1]);
  Tranload(m1Cache[0],(xload1+64),0,reg_m1[2]);
  Tranload(m1Cache[0],(xload1+64),1,reg_m1[3]);

  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 16));

  *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
  *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 64));

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 8;
      m1_load+=2;
      m2_load+=2*N;

      if(sm < K){
          //load next tile from global to reg
          reg_m2 = *(m2_load);
          reg_m1[0] = *(m1_load);
          reg_m1[1] = *(m1_load+1);
          reg_m1[2] = *(m1_load + 16*K);
          reg_m1[3] = *(m1_load + 16*K+1);
          
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<8;++x){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 16));

          *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
          *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 64));

          Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
          Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
          Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
          Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);

          ind^=1;
      }

      //load next tile to shared
      Tranload(m1Cache[write_idx],xload1,0,reg_m1[0]);
      Tranload(m1Cache[write_idx],xload1,1,reg_m1[1]);
      Tranload(m1Cache[write_idx],(xload1+64),0,reg_m1[2]);
      Tranload(m1Cache[write_idx],(xload1+64),1,reg_m1[3]);
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 16));

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
  output+=16;
  Store4x4(sum,1,output,0,N);
  output+=64*N-16;
  Store4x4(sum,2,output,0,N);
  output+=16;
  Store4x4(sum,3,output,0,N);

}

__global__ void GEMM_batch_64_64x64(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;
  int tx = tid%8;//8
  int ty = tid/8;//8

  int xload1 = tid;
  // int yload1 = tid/128;

  int xload2 = tid%16;//16
  int yload2 = tid/16;//4

  // int a = tx*4 + ty*4*N;

  m1= &m1[Batch*M*K + mx*64*K];
  m2= &m2[Batch*N*K + my*64];
  output= &output[Batch*M*N + mx*64*N + my*64 + tx*4 + ty*4*N];

  __shared__ float m1Cache[2][8][64];
  __shared__ float m2Cache[2][8][64];
  
  float4 sum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2][8] = {{0}};
  float in2[2][8] = {{0}};

  float4 reg_m1[2];
  float4 reg_m2[2];

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1[0] = *(m1_load);
  reg_m1[1] = *(m1_load+1);
  reg_m2[0] = *(m2_load);
  reg_m2[1] = *(m2_load+N);

  Tranload(m1Cache[0],xload1,0,reg_m1[0]);
  Tranload(m1Cache[0],xload1,1,reg_m1[1]);
  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2[0];
  *((float4 *)(m2Cache[0][yload2+4] + xload2*4)) = reg_m2[1];

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 32));

  *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
  *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 32));

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 8;
      m1_load+=2;
      m2_load+=2*N;
      if(sm < K){
          //load next tile from global to reg
          reg_m1[0] = *(m1_load);
          reg_m1[1] = *(m1_load + 1);
          reg_m2[0] = *(m2_load);
          reg_m2[1] = *(m2_load + N);
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<8;++x){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          *((float4 *)(in2[ind]+4)) = *((float4 *)(m2Cache[load_idx][x] + 4*tx + 32));

          *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));
          *((float4 *)(in1[ind]+4)) = *((float4 *)(m1Cache[load_idx][x] + 4*ty + 32));

          Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);
          Bccum(sum,1,in1[ind^1],0,in2[ind^1],4);
          Bccum(sum,2,in1[ind^1],4,in2[ind^1],0);
          Bccum(sum,3,in1[ind^1],4,in2[ind^1],4);

          ind^=1;
      }

      //load next tile to shared
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2[0];
      *((float4 *)(m2Cache[write_idx][yload2+4] + xload2*4)) = reg_m2[1];
      Tranload(m1Cache[write_idx],xload1,0,reg_m1[0]);
      Tranload(m1Cache[write_idx],xload1,1,reg_m1[1]);

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx + 32));

      *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));
      *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty + 32));

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
  output+=32;
  Store4x4(sum,1,output,0,N);
  output+=32*N-32;
  Store4x4(sum,2,output,0,N);
  output+=32;
  Store4x4(sum,3,output,0,N);

}

__global__ void GEMM_batch_64_32x32(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;
  int tx = tid%8;//8
  int ty = tid/8;//8

  int xload1 = tid%32;//32
  int yload1 = tid/32;//2

  int xload2 = tid%8;//8
  int yload2 = tid/8;//8

  // int a = tx*4 + ty*4*N;

  m1= &m1[Batch*M*K + mx*32*K];
  m2= &m2[Batch*N*K + my*32];
  output= &output[Batch*M*N + mx*32*N + my*32 + tx*4 + ty*4*N];

  __shared__ float m1Cache[2][8][32];
  __shared__ float m2Cache[2][8][32];
  
  float4 sum[4] = {make_float4(0.f,0.f,0.f,0.f)};
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2][4] = {{0}};
  float in2[2][4] = {{0}};

  float4 reg_m1;
  float4 reg_m2;

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K + (yload1)*4);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1 = *(m1_load);
  reg_m2 = *(m2_load);

  Tranload(m1Cache[0],xload1,yload1,reg_m1);
  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  // *((float4 *)(in2[0]+4)) = *((float4 *)(m2Cache[0][0] + 4*tx + 32));

  *((float4 *)(in1[0])) = *((float4 *)(m1Cache[0][0] + 4*ty));
  // *((float4 *)(in1[0]+4)) = *((float4 *)(m1Cache[0][0] + 4*ty + 32));

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 8;

      m1_load += 2;
      m2_load += 2*N;
      if(sm < K){
          reg_m1 = *(m1_load);
          reg_m2 = *(m2_load);
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<8;++x){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          *((float4 *)(in1[ind])) = *((float4 *)(m1Cache[load_idx][x] + 4*ty));

          Bccum(sum,0,in1[ind^1],0,in2[ind^1],0);

          ind^=1;
      }

      //load next tile to shared
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
      Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      *((float4 *)(in1[0])) = *((float4 *)(m1Cache[load_idx^1][0] + 4*ty));

     
      Bccum(sum,0,in1[1],0,in2[1],0);

  }while(sm<K);

  //write back
  Store4x4(sum,0,output,0,N);

}

__global__ void GEMM_batch_64_16x16(int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
  int K = chn;

  int mx = blockIdx.x;
  int my = blockIdx.y;
  int Batch = blockIdx.z;
  
  int tid = threadIdx.x;
  int tx = tid%4;//4
  int ty = tid/4;//16

  int xload1 = tid%16;//16
  int yload1 = tid/16;//4

  int xload2 = tid%4;//4
  int yload2 = tid/4;//16

  // int a = tx*4 + ty*4*N;

  m1= &m1[Batch*M*K + mx*16*K];
  m2= &m2[Batch*N*K + my*16];
  output= &output[Batch*M*N + mx*16*N + my*16 + tx*4 + ty*N];

  __shared__ float m1Cache[2][16][16];
  __shared__ float m2Cache[2][16][16];
  
  float4 sum = make_float4(0.f,0.f,0.f,0.f);
  // float4 Rsum[16] = {make_float4(0.f,0.f,0.f,0.f)};
  float in1[2] = {0};
  float in2[2][4] = {{0}};

  float4 reg_m1;
  float4 reg_m2;

  //load first tile to m1 and m2
  float4 * m1_load =  (float4 *)(m1 + (xload1)*K + (yload1)*4);
  float4 * m2_load =  (float4 *)(m2 + (xload2)*4 + (yload2)*N);

  // load to shared
  reg_m1 = *(m1_load);
  reg_m2 = *(m2_load);

  Tranload(m1Cache[0],xload1,yload1,reg_m1);
  *((float4 *)(m2Cache[0][yload2] + xload2*4)) = reg_m2;

  __syncthreads();

  //load first data to reg
  *((float4 *)(in2[0])) = *((float4 *)(m2Cache[0][0] + 4*tx));
  in1[0] = m1Cache[0][0][ty];

  int sm = 0;
  int write_idx = 1;

  do{
      sm += 16;
      m1_load += 4;
    //   m2_load += 4*N;
      if(sm < K){
          //load next tile from global to reg
          reg_m1 = *(m1_load);
          reg_m2 = *(m2_load + sm*N/4);
      }
      //load shared to reg
      int load_idx = write_idx ^ 1; 
      int ind = 1;

      #pragma unroll
      for(int x=1;x<16;x++){
          *((float4 *)(in2[ind])) = *((float4 *)(m2Cache[load_idx][x] + 4*tx));
          in1[ind] = m1Cache[load_idx][x][ty];

          sum.x += in1[ind^1]*in2[ind^1][0];
          sum.y += in1[ind^1]*in2[ind^1][1];
          sum.z += in1[ind^1]*in2[ind^1][2];
          sum.w += in1[ind^1]*in2[ind^1][3];

          ind^=1;
      }

      //load next tile to shared
      *((float4 *)(m2Cache[write_idx][yload2] + xload2*4)) = reg_m2;
      Tranload(m1Cache[write_idx],xload1,yload1,reg_m1);

      __syncthreads();
  
      write_idx ^= 1;

      //load shared to reg
      *((float4 *)(in2[0])) = *((float4 *)(m2Cache[load_idx^1][0] + 4*tx));
      in1[0] = m1Cache[load_idx^1][0][ty];

      sum.x += in1[1]*in2[1][0];
      sum.y += in1[1]*in2[1][1];
      sum.z += in1[1]*in2[1][2];
      sum.w += in1[1]*in2[1][3];

  }while(sm<K);

  //write back
  *((float4 *)(output)) = sum;

}

void wrapedGEMM_KMKN(int Batch, int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    // int blockx = (M+127)/128;
    // int blocky = (N+127)/128;
    // GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, Batch),dim3(256,1,1) >>>(M,N,chn,1,m1,m2,0,output);
    
    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    int nInput = Batch * MSize * KSize;
    int nFilter = Batch * KSize * NSize;
    int nOutput = Batch * MSize * NSize;

    float *matrix1, *matrix2, *matrix3;

    cudaMalloc((void **) &matrix1, nInput<<2);
    cudaMalloc((void **) &matrix2, nFilter<<2);
    cudaMalloc((void **) &matrix3, nOutput<<2);

    // The supposed program routine
    for(int j=0;j<Batch;j++){
        for(int i=0; i<chn; i++) {
            cudaMemcpy((matrix1 + MSize * i + j* MSize*KSize), (m1 + M * i + j*M*chn), M<<2, cudaMemcpyDeviceToDevice);
            cudaMemcpy((matrix2 + NSize * i + j* MSize*KSize), (m2 + N * i + j*M*chn), N<<2, cudaMemcpyDeviceToDevice);
        }
    }
    
    cudaMemcpy(matrix1, m1, M*chn*Batch,cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2, m2, N*chn*Batch,cudaMemcpyDeviceToDevice);
    
    //for GEMM
    int blockx = (MSize+127)/128;
    int blocky = (NSize+127)/128;

    GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, Batch),dim3(256,1,1) >>>(MSize,NSize,KSize,1,matrix1,matrix2,0,matrix3);

    // The supposed program routine
    for (int j=0; j<Batch; j++) {
        for (int i=0; i<M; i++) {
            cudaMemcpy((output+N*i + j*M*N), (matrix3 + NSize*i + MSize*NSize * j), N<<2, cudaMemcpyDeviceToDevice);
        }
    }

    cudaMemcpy(output, matrix3, M*N*Batch,cudaMemcpyDeviceToDevice);

    cudaFree(matrix1);
    cudaFree(matrix2);
    cudaFree(matrix3);
}

/*
void wrapedTest(int Batch, int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output) {
    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    int nInput = MSize*KSize;
    int nFilter = KSize*NSize;
    int nOutput = MSize*NSize;

    float *matrix1, *matrix2, *matrix3;

    cudaMalloc((void **) &matrix1, nInput<<2);
    cudaMalloc((void **) &matrix2, nFilter<<2);
    cudaMalloc((void **) &matrix3, nOutput<<2);

    for(int i=0; i<chn; i++) {
        cudaMemcpy((matrix1 + MSize * i), (m1 + M * i), M<<2, cudaMemcpyDeviceToDevice);
        cudaMemcpy((matrix2 + NSize * i), (m2 + N * i), N<<2, cudaMemcpyDeviceToDevice);
    }
    // cudaMemcpy(matrix1, m1, (M*chn)<<2, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(matrix2, m2, (N*chn)<<2, cudaMemcpyDeviceToDevice);

    // printf("nInput:%d\tnFilter:%d\tnOutput:%d\n",nInput,nFilter,nOutput);

    //for GEMM
    int blockx = (MSize+127)/128;
    int blocky = (NSize+127)/128;

    GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, Batch),dim3(256,1,1) >>>(MSize,NSize,KSize,1,matrix1,matrix2,0,matrix3);

    // float *test1 = (float *)malloc(nInput*sizeof(float));
    // float *test2 = (float *)malloc(nFilter*sizeof(float));
    // float *test3 = (float *)malloc(nOutput*sizeof(float));
    // cudaMemcpy(test1, matrix1, nInput<<2, cudaMemcpyDeviceToHost);
    // cudaMemcpy(test2, matrix2, nFilter<<2, cudaMemcpyDeviceToHost);
    // cudaMemcpy(test3, matrix3, nOutput<<2, cudaMemcpyDeviceToHost);
    // // cudaMemcpy(test1, matrix1, nInput<<2, cudaMemcpyDeviceToHost);
    // // cudaMemcpy(test2, matrix2, nFilter<<2, cudaMemcpyDeviceToHost);
    // printf("Test5:\n 1:%f\t 2:%f\t 3:%f\n",test1[0],test2[0],test3[0]);

    for (int i=0; i<M; i++) {
        cudaMemcpy((output+N*i), (matrix3 + NSize*i), N <<2, cudaMemcpyDeviceToHost);
    }

    cudaFree(matrix1);
    cudaFree(matrix2);
    cudaFree(matrix3);
}
*/

void wrapedGEMM_KMKN_test(int Batch, int M, int N, int chn, float alpha, float *m1, float *m2, float beta, float *output){
    bool MCheck = ( M % 128 == 0 )? true: false;
    bool NCheck = ( N % 128 == 0 )? true: false;
    bool KCheck = ( chn % 8 == 0 )? true: false;

    int MSize = MCheck? M: ((M/128 +1) * 128);
    int NSize = NCheck? N: ((N/128 +1) * 128);
    int KSize = KCheck? chn: ((chn/8 +1) * 8);

    float *matrix1 = m1;
    float *matrix2 = m2;
    float *matrix3 = (float *)malloc(MSize*NSize*sizeof(float));

    if (!(MCheck && NCheck && KCheck)){
        matrix1 = (float *)malloc(MSize * KSize * sizeof(float));
        matrix2 = (float *)malloc(KSize * NSize * sizeof(float));
        // for(int i=0; i<M; i++){
        //     memcpy((matrix1 + KSize * i), (m1 + chn * i), chn<<2);
        // }
        // for(int j=0; j<N; j++){
        //     memcpy((matrix2 + KSize * j), (m2 + chn * j), chn<<2);
        // }
        for(int i=0; i<chn; i++){
            memcpy((matrix1 + MSize * i), (m1 + M * i), M<<2);
        }
        for(int j=0; j<chn; j++){
            memcpy((matrix2 + NSize * j), (m2 + M * j), N<<2);
        }
    }

    int nInput = MSize*KSize;
    int nFilter = KSize*NSize;
    int nOutput = MSize*NSize;

    float *input_gpu,*filter_gpu,*output_gpu;
    
    cudaMalloc((void **) &input_gpu, nInput<<2);
    cudaMalloc((void **) &filter_gpu, nFilter<<2);
    cudaMalloc((void **) &output_gpu, nOutput<<2);

    cudaMemcpy(input_gpu, matrix1, nInput<<2, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, matrix2, nFilter<<2, cudaMemcpyHostToDevice);
    
    // //for GEMM
    int blockx = (MSize+127)/128;
    int blocky = (NSize+127)/128;

     // if((M<128)|(M%128)!=0){
    //     int min = lcm(128,(blockn*blockn))/(blockn*blockn);
    //     printf("Batch should be mutiple of %d\n",min);
    //     return 0;
    // }

    float avetime =0;

    cudaEvent_t start1,stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    //warm up
    // wino_kernel_trans_chwn<<<dim3(chn,1) , dim3(chn,1) >>>(filter_gpu , filter_gpu_tran);// no need batch
    GEMM_batch_256_128x128_KMKN<<<dim3(blocky,blockx,1),dim3(256,1,1) >>>(MSize,NSize,KSize,1,input_gpu,filter_gpu,0,output_gpu);

    cudaEventRecord(start1, NULL);
    for(int x=0;x<10;x++){
        GEMM_batch_256_128x128_KMKN<<<dim3(blocky, blockx, Batch),dim3(256,1,1) >>>(MSize,NSize,KSize,1,input_gpu,filter_gpu,0,output_gpu);
    }
    cudaEventRecord(stop1, NULL);


    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);

    cudaEventElapsedTime(&avetime, start1, stop1);
    
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    avetime = avetime/10;
    double FLOP = M*N*(chn*2-1)*1.0e-9;
    double gflops = (FLOP/avetime);
    printf("time:%lf us TFLOPS:%lf \n", (avetime*1000), gflops);

    cudaMemcpy(matrix3, output_gpu, nOutput <<2, cudaMemcpyDeviceToHost);

    for (int i=0; i<M; i++) {
        memcpy((output+N*i), (matrix3 + NSize*i), N<<2);
    }
        
    cudaFree(input_gpu);
    cudaFree(output_gpu);
    cudaFree(filter_gpu);
}