__global__ void wino_input_trans_nhwc(int side, float * pInputs, float * pOutputs){
    int chn = threadIdx.x;  // total chn is is 512
    int row = blockIdx.x; // total row size is 4
    int col = blockIdx.y; // total col size is 4
    int batch = blockIdx.z;//total is 1

    int Inside = side;
    int deep = blockDim.x;
    int Nbatch = gridDim.z;
    int M = 1 + (Inside-6)/4;

    float Mread[6][6];

    int m = chn + row*4*deep + col*Inside*4*deep + batch*Inside*Inside*deep;

    for(int i=0;i<6;i++){
        for(int k=0;k<6;k++){
            Mread[i][k] = pInputs[m + k*deep + i*Inside*deep];           
        }
    }
    
    // 16 is the location, the tilesize
    int a = batch*M*M*deep + chn + row*deep + col*M*deep;

    float Atd[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = Nbatch*M*M*deep;

    for(int i=0;i<6;i++){
        pOutputs[a + i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[a + i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[a + i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[a + i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[a + i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[a + i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}

__global__ void wino_input_trans_nhwc_refact(int side, float * pInputs, float * pOutputs){
    int tid  = threadIdx.x;  // total chn is is 512, chn
    int bidx = blockIdx.x; // total row size is 4, row
    int bidy = blockIdx.y; // total col size is 4, col
    int bidz = blockIdx.z;//total is 1, batch

    int Inside = side;
    int totalChn = blockDim.x;
    int numOfBatch =  gridDim.z;
    int blockn = 1 + (Inside-6)/4;

    float Mread[6][6];

    int input_loc = tidx + bidx*4*totalChn + bidy*Inside*4*totalChn + bidz*Inside*Inside*totalChn;

    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            Mread[i][j] = pInputs[input_loc + j*totalChn + i*Inside*totalChn];           
        }
    }
    
    // 16 is the location, the tilesize
    int output_loc = bidz*blockn*blockn*totalChn + tidx + bidx*totalChn + bidy*blockn*totalChn;

    float Atd[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
        Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
        Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
        Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
        Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
        Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
    }

    int size = numOfBatch*blockn*blockn*totalChn;

    for(int i=0;i<6;i++){
        pOutputs[output_loc + i*6*size] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
        pOutputs[output_loc + i*6*size + size] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 2*size] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 3*size] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 4*size] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
        pOutputs[output_loc + i*6*size + 5*size] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
    }
}

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



__global__ void wino_input_trans_nhwc_new(int side, float * pInputs, float * pOutputs){
    int Inside = side;
    int deep = blockDim.x*gridDim.y;
    int Nbatch = gridDim.z;
    int M = 1 + (Inside-6)/4;

    int chn = threadIdx.x; // each block deal with 64 chn
    int ty = threadIdx.y;  // use to parallem the trans 6

    int row = blockIdx.x%M; // total row size
    int col = blockIdx.x/M; // total col size

    int s_chn = blockIdx.y; //spilit chn (chn/64)
    int batch = blockIdx.z;// number of graph

    __shared__ float Mread[6][6][64];
    __shared__ float Atd[6][6][64];

    int m = chn + s_chn*64 + row*4*deep + col*Inside*4*deep + batch*Inside*Inside*deep + ty*deep;

    for(int i=0;i<6;i++){
        Mread[i][ty][chn] = pInputs[m + i*Inside*deep];           
    }
    __syncthreads();

    int a = batch*M*M*deep + chn + s_chn*64 + row*deep + col*M*deep;

    Atd[0][ty][chn] = 4*Mread[0][ty][chn] - 5*Mread[2][ty][chn] + Mread[4][ty][chn];
    Atd[1][ty][chn] = -4*Mread[1][ty][chn] -4*Mread[2][ty][chn] + Mread[3][ty][chn] + Mread[4][ty][chn];
    Atd[2][ty][chn] = 4*Mread[1][ty][chn] -4*Mread[2][ty][chn] - Mread[3][ty][chn] + Mread[4][ty][chn];
    Atd[3][ty][chn] = -2*Mread[1][ty][chn] - Mread[2][ty][chn] + 2*Mread[3][ty][chn] + Mread[4][ty][chn];
    Atd[4][ty][chn] = 2*Mread[1][ty][chn] - Mread[2][ty][chn] - 2*Mread[3][ty][chn] + Mread[4][ty][chn];
    Atd[5][ty][chn] = 4*Mread[1][ty][chn] - 5*Mread[3][ty][chn] + Mread[5][ty][chn];

    __syncthreads();
    
    int size = Nbatch*M*M*deep;

    pOutputs[a + ty*6*size] = 4*Atd[ty][0][chn] - 5*Atd[ty][2][chn] + Atd[ty][4][chn];
    pOutputs[a + ty*6*size + size] = -4*Atd[ty][1][chn] - 4*Atd[ty][2][chn] + Atd[ty][3][chn] + Atd[ty][4][chn];
    pOutputs[a + ty*6*size + 2*size] = 4*Atd[ty][1][chn] - 4*Atd[ty][2][chn] - Atd[ty][3][chn] + Atd[ty][4][chn];
    pOutputs[a + ty*6*size + 3*size] = -2*Atd[ty][1][chn] - Atd[ty][2][chn] + 2*Atd[ty][3][chn] + Atd[ty][4][chn];
    pOutputs[a + ty*6*size + 4*size] = 2*Atd[ty][1][chn] - Atd[ty][2][chn] - 2*Atd[ty][3][chn] + Atd[ty][4][chn];
    pOutputs[a + ty*6*size + 5*size] = 4*Atd[ty][1][chn] - 5*Atd[ty][3][chn] + Atd[ty][5][chn];
    __syncthreads();
}

__global__ void wino_input_trans_nhwc_new_refact(int side, float * pInputs, float * pOutputs){
    int blockn = M = 1 + (Inside-6)/4;
    
    int tidx = chn = threadIdx.x; // each block deal with 64 chn
    int tidy = ty = threadIdx.y;  // use to parallem the trans 6

    int blocknx = row = blockIdx.x%blockn; // total row size, row
    int blockny = col = blockIdx.x/blockn; // total col size, col


    int bidy = s_chn = blockIdx.y; //spilit chn (chn/64)
    int bidz = batch = blockIdx.z;// number of graph
    
    
    int Inside = side;
    int totalChn = deep = blockDim.x*gridDim.y;
    int numOfBatch = Nbatch = gridDim.z;
    

    __shared__ float Mread[6][6][64];
    __shared__ float Atd[6][6][64];

    int input_loc = tidx + bidy*64 + blocknx*4*totalChn + blockny*Inside*4*totalChn + bidz*Inside*Inside*totalChn + tidy*totalChn;

    for(int i=0;i<6;i++){
        Mread[i][tidy][tidx] = pInputs[input_loc + i*Inside*totalChn];           
    }
    __syncthreads();

    int output_loc = bidz*blockn*blockn*totalChn + tidx + bidy*64 + blocknx*totalChn + blockny*blockn*totalChn;

    Atd[0][tidy][tidx] =  4*Mread[0][tidy][tidx] - 5*Mread[2][tidy][tidx] +   Mread[4][tidy][tidx];
    Atd[1][tidy][tidx] = -4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] +   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[2][tidy][tidx] =  4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] -   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[3][tidy][tidx] = -2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] + 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[4][tidy][tidx] =  2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] - 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[5][tidy][tidx] =  4*Mread[1][tidy][tidx] - 5*Mread[3][tidy][tidx] +   Mread[5][tidy][tidx];

    __syncthreads();
    
    int size = numOfBatch * blockn * blockn * totalChn;

    pOutputs[output_loc + tidy*6*size]          =  4*Atd[tidy][0][tidx] - 5*Atd[tidy][2][tidx] +   Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + size]   = -4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] +   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 2*size] =  4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] -   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 3*size] = -2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] + 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 4*size] =  2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] - 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 5*size] =  4*Atd[tidy][1][tidx] - 5*Atd[tidy][3][tidx] +   Atd[tidy][5][tidx];
    __syncthreads();
}

__global__ void wino_input_trans_nhwc_new_refact2(int side, float * pInputs, float * pOutputs){
    int Inside = side;
    int blockn = 1+ (Inside-2)/4;
    
    int tidx = threadIdx.x; // each for 1 chn and total size of 64 threads.
    int tidy = threadIdx.y; // up to parallelism of 6;

    int bidx = blockIdx.x;  // total number of blockn
    int blocknx = bidx%blockn;
    int blockny = bidx/blockn;

    int bidy = blockIdx.y;  // each one corresponding for 64 chn, so equal to (totolChn/64)
    int bidz = blockIdx.z;  // batch

    totalChn = gridDim.y*64;
    numOfBatch = gridDim.z;

    pInputs  = &pInputs [tidx + tidy*totalChn + bidy*64 + blocknx *4*totalChn + blockny*4*Inside*totalChn + bidz*Inside*Inside*totalChn];
    pOutputs = &pOutputs[tidx + bidy*64 + bidz*MSize + blocknx*numOfBatch*MSize + blockny*numOfBatch*MSize*blockn];

    float Mread[6][6] = {{0}};

    for(int i=0;i<6;i++){
        Mread[i][tidy][tidx] = pInputs[input_loc + i*Inside*totalChn];           
    }

    __shared__ float Mread[6][6][64];
    __shared__ float Atd[6][6][64];

    Atd[0][tidy][tidx] =  4*Mread[0][tidy][tidx] - 5*Mread[2][tidy][tidx] +   Mread[4][tidy][tidx];
    Atd[1][tidy][tidx] = -4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] +   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[2][tidy][tidx] =  4*Mread[1][tidy][tidx] - 4*Mread[2][tidy][tidx] -   Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[3][tidy][tidx] = -2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] + 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[4][tidy][tidx] =  2*Mread[1][tidy][tidx] -   Mread[2][tidy][tidx] - 2*Mread[3][tidy][tidx] + Mread[4][tidy][tidx];
    Atd[5][tidy][tidx] =  4*Mread[1][tidy][tidx] - 5*Mread[3][tidy][tidx] +   Mread[5][tidy][tidx];

    int size = numOfBatch * blockn * blockn * totalChn;
    
    pOutputs[output_loc + tidy*6*size]          =  4*Atd[tidy][0][tidx] - 5*Atd[tidy][2][tidx] +   Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + size]   = -4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] +   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 2*size] =  4*Atd[tidy][1][tidx] - 4*Atd[tidy][2][tidx] -   Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 3*size] = -2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] + 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 4*size] =  2*Atd[tidy][1][tidx] -   Atd[tidy][2][tidx] - 2*Atd[tidy][3][tidx] + Atd[tidy][4][tidx];
    pOutputs[output_loc + tidy*6*size + 5*size] =  4*Atd[tidy][1][tidx] - 5*Atd[tidy][3][tidx] +   Atd[tidy][5][tidx];
    __syncthreads();
}


__global__ void wino_kernel_trans_nhwc(float * pInputs, float * pOutputs){
    int num = threadIdx.x;  // total num is 512
    int chn = blockIdx.x; // total chn is 512
    int deep = blockDim.x;
            
    float Mread[3][3] = {{0}};
   
    for(int i=0;i<9;i++){ 
        Mread[i/3][i%3] = pInputs[9*chn + num*9*deep + i];               
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

    int a = deep*chn + num;
    int size = deep*deep;

    for(int i=0;i<6;i++){
        pOutputs[a + i*6*size] = Gg[i][0]/4;
        pOutputs[a + i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 5*size] = Gg[i][2];
    }
}

__global__ void wino_invers_nhwc(int relu_pooling, int oside, float * pInputs, float * pOutputs){
    int tX = blockIdx.x;  // total Inx is is 4
    int tY = blockIdx.y;  // total Iny is is 4
    int Batch = blockIdx.z;//1

    int deep = blockDim.x;
    int locnum = gridDim.x*gridDim.x;

    int chn = threadIdx.x; // total chn size is 512

    float Mread[6][6];
    float outreg[4][4];
    float rp_reg[2][2];

    int load = Batch*locnum*deep + tX*deep + tY*gridDim.x*deep + chn;
    int size = gridDim.z*locnum*deep;

    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            Mread[i][j] = pInputs[load + j*size + i*6*size];
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

    for(int i=0;i<4;i++){
        outreg[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        outreg[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        outreg[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        outreg[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    if(relu_pooling == 0){
        int a = chn + tX*4*deep + tY*4*oside*deep + Batch*oside*oside*deep;
        for(int i=0;i<4;i++){
            pOutputs[a + i*oside*deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + 2*deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + 3*deep] = outreg[i][0];
        }
    }else{
        int a = chn + tX*2*deep + tY*2*oside/2*deep + Batch*oside/2*oside/2*deep;
        //relu
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                if(outreg[i][j]<=0)
                    outreg[i][j]=0;
            }
        }

        //max pooling
        rp_reg[0][0] = max(outreg[0][0],max(outreg[0][1],max(outreg[1][0],outreg[1][1])));
        rp_reg[0][1] = max(outreg[0][2],max(outreg[0][3],max(outreg[1][2],outreg[1][3])));
        rp_reg[1][0] = max(outreg[2][0],max(outreg[2][1],max(outreg[3][0],outreg[3][1])));
        rp_reg[1][1] = max(outreg[2][2],max(outreg[2][3],max(outreg[3][2],outreg[3][3])));

        for(int i=0;i<2;i++){
            pOutputs[a + i*oside/2*deep] = rp_reg[i][0];
            pOutputs[a + i*oside/2*deep + deep] = rp_reg[i][1];
        }
    }
    
}




__global__ void wino_input_trans_nchw(int inside, float * pInputs, float * pOutputs){
    int tid = threadIdx.x;
    int chn = blockIdx.x;//input chn 
    int N = blockIdx.z;//input number of graph
    int tilenum = 1 + (inside-6)/4;

    extern __shared__ float inCache[];//use to load input graph// 4 8 
    float Mread[6][6] = {{0}};//use to hold the input

    if(inside<=34){
        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*8*inside*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<8;i++){
            for(int j=0;j<2;j++){
                if(tid*4 + j*4*blockDim.x < inside*inside)
                    *((float4 *)(inCache + 4*tid +  + j*4*blockDim.x + inside*inside*i)) = *((float4 *)(pInputs + 4*tid + inside*inside*gridDim.x*i + j*4*blockDim.x));
            }
        }
        __syncthreads();

        //load the data to reg
        if(tid < tilenum*tilenum*8){
            int tidx = tid%(tilenum*tilenum);
            int tidy = tid/(tilenum*tilenum);//8

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tidx%tilenum)*4 + (tidx/tilenum)*4*inside + tidy*inside*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum*8 + chn*8*gridDim.z*tilenum*tilenum + tid];
            int size = gridDim.x*tilenum*tilenum*gridDim.z*8;
            
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
    }else if(inside <= 46){
        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*4*inside*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<4;i++){
            for(int j=0;j<2;j++){
                if(tid*4 + j*4*blockDim.x < inside*inside)
                    *((float4 *)(inCache + 4*tid +  + j*4*blockDim.x + inside*inside*i)) = *((float4 *)(pInputs + 4*tid + inside*inside*gridDim.x*i + j*4*blockDim.x));
            }
        }
        __syncthreads();

        //load the data to reg
        if(tid < tilenum*tilenum*4){
            int tidx = tid%(tilenum*tilenum);
            int tidy = tid/(tilenum*tilenum);//4

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tidx%tilenum)*4 + (tidx/tilenum)*4*inside + tidy*inside*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum*4 + chn*4*gridDim.z*tilenum*tilenum + tid];
            int size = gridDim.x*tilenum*tilenum*gridDim.z*4;
            
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

    }else if(inside <= 82){
        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*inside*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<5;i++){
            if(tid*4 + i*blockDim.x*4 < inside*inside)
                *((float4 *)(inCache + 4*tid + i*blockDim.x*4)) = *((float4 *)(pInputs + 4*tid + i*blockDim.x*4));
        }
        __syncthreads();

        //load the data to reg
        if(tid < tilenum*tilenum){

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tid%tilenum)*4 + (tid/tilenum)*4*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum + chn*gridDim.z*tilenum*tilenum + tid];
            int size = gridDim.x*tilenum*tilenum*gridDim.z;
            
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
    }else if(inside <= 162){
        int blocky = blockIdx.y;//use to spilit y direction 4 block
        int s_tile = (tilenum + 3)/4;

        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*inside*inside + blocky*4*s_tile*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<5;i++){
            if(tid*4 + i*blockDim.x*4 < (4*s_tile+2)*inside)
                *((float4 *)(inCache + 4*tid + i*blockDim.x*4)) = *((float4 *)(pInputs + 4*tid + i*blockDim.x*4));
        }
        __syncthreads();

        //load the data to reg
        if(tid < s_tile*tilenum){

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tid%tilenum)*4 + (tid/tilenum)*4*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum + chn*gridDim.z*tilenum*tilenum + tid + blocky*s_tile*tilenum];
            int size = gridDim.x*tilenum*tilenum*gridDim.z;
            
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

    }else if(inside <= 322){
        int blocky = blockIdx.y;//use to spilit y direction 4 block
        int s_tile = (tilenum + 15)/16;

        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*inside*inside + blocky*4*s_tile*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<5;i++){
            if(tid*4 + i*blockDim.x*4 < (4*s_tile+2)*inside)
                *((float4 *)(inCache + 4*tid + i*blockDim.x*4)) = *((float4 *)(pInputs + 4*tid + i*blockDim.x*4));
        }
        __syncthreads();

        //load the data to reg
        if(tid < s_tile*tilenum){

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tid%tilenum)*4 + (tid/tilenum)*4*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum + chn*gridDim.z*tilenum*tilenum + tid + blocky*s_tile*tilenum];
            int size = gridDim.x*tilenum*tilenum*gridDim.z;
            
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

    }else{
        int blocky = blockIdx.y;//use to spilit y direction 4 block
        int s_tile = (tilenum + 39)/40;

        pInputs = &pInputs[chn*inside*inside + N*gridDim.x*inside*inside + blocky*4*s_tile*inside];// locate the input position
        //load the 8 tile(input) to shared
        for(int i=0;i<5;i++){
            if(tid*4 + i*blockDim.x*4 < (4*s_tile+2)*inside)
                *((float4 *)(inCache + 4*tid + i*blockDim.x*4)) = *((float4 *)(pInputs + 4*tid + i*blockDim.x*4));
        }
        __syncthreads();

        //load the data to reg
        if(tid < s_tile*tilenum){

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = inCache[i*inside + j + (tid%tilenum)*4 + (tid/tilenum)*4*inside];
                }
            }

            //use the data to caculate
            float Atd[6][6] = {{0}};
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
                Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
                Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
                Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
                Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
            }

            pOutputs = &pOutputs[N*tilenum*tilenum + chn*gridDim.z*tilenum*tilenum + tid + blocky*s_tile*tilenum];
            int size = gridDim.x*tilenum*tilenum*gridDim.z;
            
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

    }
    
    __syncthreads();
    
}

__global__ void wino_invers_nchw(int relu_pooling, int oside, float * pInputs, float * pOutputs){
    int chn = blockIdx.x;
    int Batch = blockIdx.z;
    int tid = threadIdx.x;

    float Mread[6][6];//use to store input
    float Mout[4][4];//use to store output

    int inx = 1 + (oside-4)/4;

    if(oside <= 32){
        int size = inx*inx*gridDim.z*gridDim.x*8;//size of 1batch(total36)
        if(tid<inx*inx*8){
            int tidx = tid%(inx*inx);
            int tidy = tid/(inx*inx);

            pInputs = &pInputs[chn*inx*inx*gridDim.z*8 + Batch*8*inx*inx + tid];

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tidx%inx)*4 + (tidx/inx)*4*oside + tidy*(oside*oside*gridDim.x) + Batch*oside*oside*gridDim.x*8 + chn*oside*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tidx%inx)*2 + (tidx/inx)*2*oside/2 + tidy*(oside*oside*gridDim.x/4) + Batch*oside*oside*gridDim.x*8/4 + chn*oside*oside/4];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
            
        }
    }else if(oside <= 44){
        int size = inx*inx*gridDim.z*gridDim.x*4;//size of 1batch(total36)
        if(tid<inx*inx*4){
            int tidx = tid%(inx*inx);
            int tidy = tid/(inx*inx);

            pInputs = &pInputs[chn*inx*inx*gridDim.z*4 + Batch*4*inx*inx + tid];

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tidx%inx)*4 + (tidx/inx)*4*oside + tidy*(oside*oside*gridDim.x) + Batch*oside*oside*gridDim.x*4 + chn*oside*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tidx%inx)*2 + (tidx/inx)*2*oside/2 + tidy*(oside*oside*gridDim.x/4) + Batch*oside*oside*gridDim.x*4/4 + chn*oside*oside/4];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
        }
    }else if(oside <= 80){
        int size = inx*inx*gridDim.z*gridDim.x;//size of 1batch(total36)
        if(tid<inx*inx){
            pInputs = &pInputs[chn*inx*inx*gridDim.z + Batch*inx*inx + tid];
            

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tid%inx)*4 + (tid/inx)*4*oside + Batch*oside*oside*gridDim.x + chn*oside*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tid%inx)*2 + (tid/inx)*2*oside/2 + Batch*oside*oside*gridDim.x/4 + chn*oside*oside/4];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
        }
    }else if(oside <=160){
        //use blocky to spilit the input in y direction
        int blocky = blockIdx.y; //4 block to spilit
        int s_tile = (inx + 3)/4;

        int size = inx*inx*gridDim.z*gridDim.x;//size of 1batch(total36)
        if(tid<s_tile*inx){
            pInputs = &pInputs[chn*inx*inx*gridDim.z + Batch*inx*inx + tid + blocky*s_tile*inx];
            

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tid%inx)*4 + (tid/inx)*4*oside + Batch*oside*oside*gridDim.x + chn*oside*oside + blocky*4*s_tile*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tid%inx)*2 + (tid/inx)*2*oside/2 + Batch*oside*oside*gridDim.x/4 + chn*oside*oside/4 + blocky*s_tile*oside];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
        }
         
    }else if(oside <=320){
        //use blocky to spilit the input in y direction
        int blocky = blockIdx.y; //16 block to spilit
        int s_tile = (inx + 15)/16;

        int size = inx*inx*gridDim.z*gridDim.x;//size of 1batch(total36)
        if(tid<s_tile*inx){
            pInputs = &pInputs[chn*inx*inx*gridDim.z + Batch*inx*inx + tid + blocky*s_tile*inx];

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tid%inx)*4 + (tid/inx)*4*oside + Batch*oside*oside*gridDim.x + chn*oside*oside + blocky*4*s_tile*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tid%inx)*2 + (tid/inx)*2*oside/2 + Batch*oside*oside*gridDim.x/4 + chn*oside*oside/4 + blocky*2*s_tile/2*oside/2];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
        }
         
    }else{
        //use blocky to spilit the input in y direction
        int blocky = blockIdx.y; //16 block to spilit
        int s_tile = (inx + 39)/40;

        int size = inx*inx*gridDim.z*gridDim.x;//size of 1batch(total36)
        if(tid<s_tile*inx){
            pInputs = &pInputs[chn*inx*inx*gridDim.z + Batch*inx*inx + tid + blocky*s_tile*inx];


            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    Mread[i][j] = pInputs[j*size + i*6*size];
                }
            }

            float Atd[4][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
                Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
                Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
                Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
            }

            for(int i=0;i<4;i++){
                Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
                Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
                Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
                Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
            }
            
            if(relu_pooling==0){
                pOutputs = &pOutputs[(tid%inx)*4 + (tid/inx)*4*oside + Batch*oside*oside*gridDim.x + chn*oside*oside + blocky*4*s_tile*oside];
                //output
                *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
                *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
                *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
                *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
            }else{
                pOutputs = &pOutputs[(tid%inx)*2 + (tid/inx)*2*oside/2 + Batch*oside*oside*gridDim.x/4 + chn*oside*oside/4 + blocky*2*s_tile/2*oside/2];
                float rp_reg[2][2] = {{0}};
                //relu
                for(int i=0;i<4;i++){
                    for(int j=0;j<4;j++){
                        if(Mout[i][j]<=0)
                            Mout[i][j]=0;
                    }
                }

                //max pooling
                rp_reg[0][0] = max(Mout[0][0],max(Mout[0][1],max(Mout[1][0],Mout[1][1])));
                rp_reg[0][1] = max(Mout[0][2],max(Mout[0][3],max(Mout[1][2],Mout[1][3])));
                rp_reg[1][0] = max(Mout[2][0],max(Mout[2][1],max(Mout[3][0],Mout[3][1])));
                rp_reg[1][1] = max(Mout[2][2],max(Mout[2][3],max(Mout[3][2],Mout[3][3])));

                *((float2 *)(pOutputs)) = *((float2 *)(rp_reg[0]));
                *((float2 *)(pOutputs + oside/2)) = *((float2 *)(rp_reg[1]));
            }
        }
         
    }

    __syncthreads();
}

__global__ void wino_kernel_trans_nchw(float * pInputs, float * pOutputs){
    int num = threadIdx.x; 
    int chn = blockIdx.x; 
    int ty = blockIdx.y;
    int deep = gridDim.x;
            
    float Mread[3][3] = {{0}};
    pInputs = &pInputs[9*chn + num*9*deep + ty*(blockDim.x*9*deep)];
    pOutputs = &pOutputs[deep*chn + num + ty*blockDim.x];
    const int size = deep*deep;
   
    *((float3 *)(Mread[0])) = *((float3 *)(pInputs));
    *((float3 *)(Mread[1])) = *((float3 *)(pInputs+3));
    *((float3 *)(Mread[2])) = *((float3 *)(pInputs+6));

    pOutputs[0] = (Mread[0][0]/4)/4;
    pOutputs[size] = -(Mread[0][0]/4)/6 - (Mread[0][1]/4)/6 -(Mread[0][2]/4)/6;
    pOutputs[2*size] = -(Mread[0][0]/4)/6 + (Mread[0][1]/4)/6 -(Mread[0][2]/4)/6;
    pOutputs[3*size] = (Mread[0][0]/4)/24 + (Mread[0][1]/4)/12 + (Mread[0][2]/4)/6;
    pOutputs[4*size] = (Mread[0][0]/4)/24 - (Mread[0][1]/4)/12 + (Mread[0][2]/4)/6;
    pOutputs[5*size] = (Mread[0][2]/4);
    pOutputs[6*size] = (-Mread[0][0]/6 - Mread[1][0]/6 - Mread[2][0]/6)/4;
    pOutputs[7*size] = -(-Mread[0][0]/6 - Mread[1][0]/6 - Mread[2][0]/6)/6 - (-Mread[0][1]/6 - Mread[1][1]/6 - Mread[2][1]/6)/6 -(-Mread[0][2]/6 - Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[8*size] = -(-Mread[0][0]/6 - Mread[1][0]/6 - Mread[2][0]/6)/6 + (-Mread[0][1]/6 - Mread[1][1]/6 - Mread[2][1]/6)/6 -(-Mread[0][2]/6 - Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[9*size] = (-Mread[0][0]/6 - Mread[1][0]/6 - Mread[2][0]/6)/24 + (-Mread[0][1]/6 - Mread[1][1]/6 - Mread[2][1]/6)/12 + (-Mread[0][2]/6 - Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[10*size] = (-Mread[0][0]/6 - Mread[1][0]/6 - Mread[2][0]/6)/24 - (-Mread[0][1]/6 - Mread[1][1]/6 - Mread[2][1]/6)/12 + (-Mread[0][2]/6 - Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[11*size] = (-Mread[0][2]/6 - Mread[1][2]/6 - Mread[2][2]/6);
    pOutputs[12*size] = (-Mread[0][0]/6 + Mread[1][0]/6 - Mread[2][0]/6)/4;
    pOutputs[13*size] = -(-Mread[0][0]/6 + Mread[1][0]/6 - Mread[2][0]/6)/6 - (-Mread[0][1]/6 + Mread[1][1]/6 - Mread[2][1]/6)/6 -(-Mread[0][2]/6 + Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[14*size] = -(-Mread[0][0]/6 + Mread[1][0]/6 - Mread[2][0]/6)/6 + (-Mread[0][1]/6 + Mread[1][1]/6 - Mread[2][1]/6)/6 -(-Mread[0][2]/6 + Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[15*size] = (-Mread[0][0]/6 + Mread[1][0]/6 - Mread[2][0]/6)/24 + (-Mread[0][1]/6 + Mread[1][1]/6 - Mread[2][1]/6)/12 + (-Mread[0][2]/6 + Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[16*size] = (-Mread[0][0]/6 + Mread[1][0]/6 - Mread[2][0]/6)/24 - (-Mread[0][1]/6 + Mread[1][1]/6 - Mread[2][1]/6)/12 + (-Mread[0][2]/6 + Mread[1][2]/6 - Mread[2][2]/6)/6;
    pOutputs[17*size] = (-Mread[0][2]/6 + Mread[1][2]/6 - Mread[2][2]/6);

    pOutputs[18*size] = (Mread[0][0]/24 + Mread[1][0]/12 + Mread[2][0]/6)/4;
    pOutputs[19*size] = -(Mread[0][0]/24 + Mread[1][0]/12 + Mread[2][0]/6)/6 - (Mread[0][1]/24 + Mread[1][1]/12 + Mread[2][1]/6)/6 -(Mread[0][2]/24 + Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[20*size] = -(Mread[0][0]/24 + Mread[1][0]/12 + Mread[2][0]/6)/6 + (Mread[0][1]/24 + Mread[1][1]/12 + Mread[2][1]/6)/6 -(Mread[0][2]/24 + Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[21*size] = (Mread[0][0]/24 + Mread[1][0]/12 + Mread[2][0]/6)/24 + (Mread[0][1]/24 + Mread[1][1]/12 + Mread[2][1]/6)/12 + (Mread[0][2]/24 + Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[22*size] = (Mread[0][0]/24 + Mread[1][0]/12 + Mread[2][0]/6)/24 - (Mread[0][1]/24 + Mread[1][1]/12 + Mread[2][1]/6)/12 + (Mread[0][2]/24 + Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[23*size] = (Mread[0][2]/24 + Mread[1][2]/12 + Mread[2][2]/6);

    pOutputs[24*size] = (Mread[0][0]/24 - Mread[1][0]/12 + Mread[2][0]/6)/4;
    pOutputs[25*size] = -(Mread[0][0]/24 - Mread[1][0]/12 + Mread[2][0]/6)/6 - (Mread[0][1]/24 - Mread[1][1]/12 + Mread[2][1]/6)/6 -(Mread[0][2]/24 - Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[26*size] = -(Mread[0][0]/24 - Mread[1][0]/12 + Mread[2][0]/6)/6 + (Mread[0][1]/24 - Mread[1][1]/12 + Mread[2][1]/6)/6 -(Mread[0][2]/24 - Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[27*size] = (Mread[0][0]/24 - Mread[1][0]/12 + Mread[2][0]/6)/24 + (Mread[0][1]/24 - Mread[1][1]/12 + Mread[2][1]/6)/12 + (Mread[0][2]/24 - Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[28*size] = (Mread[0][0]/24 - Mread[1][0]/12 + Mread[2][0]/6)/24 - (Mread[0][1]/24 - Mread[1][1]/12 + Mread[2][1]/6)/12 + (Mread[0][2]/24 - Mread[1][2]/12 + Mread[2][2]/6)/6;
    pOutputs[29*size] = (Mread[0][2]/24 - Mread[1][2]/12 + Mread[2][2]/6);
    
    pOutputs[30*size] = (Mread[2][0])/4;
    pOutputs[31*size] = -(Mread[2][0])/6 - (Mread[2][1])/6 -(Mread[2][2])/6;
    pOutputs[32*size] = -(Mread[2][0])/6 + (Mread[2][1])/6 -(Mread[2][2])/6;
    pOutputs[33*size] = (Mread[2][0])/24 + (Mread[2][1])/12 + (Mread[2][2])/6;
    pOutputs[34*size] = (Mread[2][0])/24 - (Mread[2][1])/12 + (Mread[2][2])/6;
    pOutputs[35*size] = (Mread[2][2]);
}

__global__ void wino_input_trans_nchw_spilit(int inside, float * pInputs, float * pOutputs){
    int tid = threadIdx.x;
    int chn = blockIdx.x;//input chn 
    int N = blockIdx.z;//input number of graph
    int tilenum = 1 + (inside-6)/4;

    extern __shared__ float inCache[];//use to load input graph// 
    float Mread[6][6] = {{0}};//use to hold the input
    int blocky = blockIdx.y;//use to spilit y direction

    pInputs = &pInputs[chn*inside*inside + N*gridDim.x*inside*inside + blocky*4*inside];// locate the input position
    //load the 8 tile(input) to shared
    for(int i=0;i<7;i++){
        if(tid*4 + i*blockDim.x*4 < 6*inside)
            *((float4 *)(inCache + 4*tid + i*blockDim.x*4)) = *((float4 *)(pInputs + 4*tid + i*blockDim.x*4));
    }
    __syncthreads();

    //load the data to reg
    if(tid < tilenum){

        #pragma unroll
        for(int i=0;i<6;i++){
            #pragma unroll
            for(int j=0;j<6;j++){
                Mread[i][j] = inCache[i*inside + j + (tid%tilenum)*4 + (tid/tilenum)*4*inside];
            }
        }

        //use the data to caculate
        float Atd[6][6] = {{0}};
        for(int i=0;i<6;i++){
            Atd[0][i] = 4*Mread[0][i] - 5*Mread[2][i] + Mread[4][i];
            Atd[1][i] = -4*Mread[1][i] -4*Mread[2][i] + Mread[3][i] + Mread[4][i];
            Atd[2][i] = 4*Mread[1][i] -4*Mread[2][i] - Mread[3][i] + Mread[4][i];
            Atd[3][i] = -2*Mread[1][i] - Mread[2][i] + 2*Mread[3][i] + Mread[4][i];
            Atd[4][i] = 2*Mread[1][i] - Mread[2][i] - 2*Mread[3][i] + Mread[4][i];
            Atd[5][i] = 4*Mread[1][i] - 5*Mread[3][i] + Mread[5][i];
        }

        pOutputs = &pOutputs[N*tilenum*tilenum + chn*gridDim.z*tilenum*tilenum + tid + blocky*tilenum];
        int size = gridDim.x*tilenum*tilenum*gridDim.z;
        
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
}

__global__ void wino_invers_nchw_spilit(int oside, float * pInputs, float * pOutputs){
    int chn = blockIdx.x;
    int Batch = blockIdx.z;
    int tid = threadIdx.x;

    float Mread[6][6];//use to store input
    float Mout[4][4];//use to store output

    int inx = 1 + (oside-4)/4;

    //use blocky to spilit the input in y direction
    int blocky = blockIdx.y; //block to spilit

    int size = inx*inx*gridDim.z*gridDim.x;//size of 1batch(total36)
    if(tid<inx){
        pInputs = &pInputs[chn*inx*inx*gridDim.z + Batch*inx*inx + tid + blocky*inx];
        pOutputs = &pOutputs[(tid%inx)*4 + (tid/inx)*4*oside + Batch*oside*oside*gridDim.x + chn*oside*oside + blocky*4*oside];

        #pragma unroll
        for(int i=0;i<6;i++){
            #pragma unroll
            for(int j=0;j<6;j++){
                Mread[i][j] = pInputs[j*size + i*6*size];
            }
        }

        float Atd[4][6] = {{0}};

        for(int i=0;i<6;i++){
            Atd[0][i] = Mread[0][i] + Mread[1][i] + Mread[2][i] + Mread[3][i] + Mread[4][i];
            Atd[1][i] = Mread[1][i] - Mread[2][i] + 2*Mread[3][i] - 2*Mread[4][i];
            Atd[2][i] = Mread[1][i] + Mread[2][i] + 4*Mread[3][i] + 4*Mread[4][i];
            Atd[3][i] = Mread[1][i] - Mread[2][i] + 8*Mread[3][i] - 8*Mread[4][i] + Mread[5][i];
        }

        for(int i=0;i<4;i++){
            Mout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
            Mout[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
            Mout[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
            Mout[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        }
        
        //output
        *((float4 *)(pOutputs)) = *((float4 *)(Mout[0]));
        *((float4 *)(pOutputs + oside)) = *((float4 *)(Mout[1]));
        *((float4 *)(pOutputs + 2*oside)) = *((float4 *)(Mout[2]));
        *((float4 *)(pOutputs + 3*oside)) = *((float4 *)(Mout[3]));
    }

    __syncthreads();
}



__global__ void wino_input_trans_chwn(int side, float * pInputs, float * pOutputs){
    int row = blockIdx.x; 
    int col = blockIdx.y; 
    int chn = blockIdx.z; 

    int batch = threadIdx.x; 

    float Mread[6][6];

    int deep = blockDim.x;
    int Inside = side;

    pInputs = &pInputs[chn*Inside*Inside*deep + batch + row*4*deep + col*4*Inside*deep];
    pOutputs = &pOutputs[batch + row*deep + col*gridDim.x*deep + chn*gridDim.x*gridDim.x*deep];

    #pragma unroll
    for(int i=0;i<6;i++){
        #pragma unroll
        for(int k=0;k<6;k++){
            Mread[i][k] = pInputs[k*deep + i*Inside*deep];           
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

    int size = deep*gridDim.x*gridDim.x*gridDim.z;

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

__global__ void wino_kernel_trans_chwn(float * pInputs, float * pOutputs){
    int chn = blockIdx.x; // total in_chn 
    int num = threadIdx.x;  // total num of filter(chn_out)

    int chn_out = blockDim.x;
            
    float Mread[3][3] = {{0}};
   
    for(int i=0;i<3;i++){ 
        for(int j=0;j<3;j++){
            Mread[i][j] = pInputs[num + i*3*chn_out + j*chn_out + chn*9*chn_out];      
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

    int a = chn_out*chn + num;
    int size = chn_out*chn_out;

    for(int i=0;i<6;i++){
        pOutputs[a + i*6*size] = Gg[i][0]/4;
        pOutputs[a + i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 5*size] = Gg[i][2];
    }
}

__global__ void wino_kernel_trans_chwn_new(float * pInputs, float * pOutputs){
    int chn = blockIdx.x; // total in_chn 
    int ty = blockIdx.y;
    int num = threadIdx.x;  // total num of filter(chn_out)

    int chn_out = blockDim.x*gridDim.y;
            
    float Mread[3][3] = {{0}};
   
    for(int i=0;i<3;i++){ 
        for(int j=0;j<3;j++){
            Mread[i][j] = pInputs[num + i*3*chn_out + j*chn_out + chn*9*chn_out + ty*blockDim.x];      
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

    int a = chn_out*chn + num + ty*blockDim.x;
    int size = chn_out*chn_out;

    for(int i=0;i<6;i++){
        pOutputs[a + i*6*size] = Gg[i][0]/4;
        pOutputs[a + i*6*size + size] = -Gg[i][0]/6 - Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 2*size] = -Gg[i][0]/6 + Gg[i][1]/6 -Gg[i][2]/6;
        pOutputs[a + i*6*size + 3*size] = Gg[i][0]/24 + Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 4*size] = Gg[i][0]/24 - Gg[i][1]/12 + Gg[i][2]/6;
        pOutputs[a + i*6*size + 5*size] = Gg[i][2];
    }
}

__global__ void wino_invers_chwn(int relu_pooling, int oside, float * pInputs, float * pOutputs){
    int chn = blockIdx.x;
    int tX = blockIdx.y; 
    int tY = blockIdx.z; 
    int Batch = threadIdx.x; 

    float Mread[6][6];
    float outreg[4][4];
    float rp_reg[2][2];


    int size = gridDim.y*gridDim.y*blockDim.x*gridDim.x;
    int deep = blockDim.x;

    pInputs = &pInputs[tX*blockDim.x + tY*gridDim.y*blockDim.x + chn*gridDim.y*gridDim.y*blockDim.x + Batch];
    // pOutputs = &pOutputs[tX*4*blockDim.x + tY*4*oside*blockDim.x + chn*oside*oside*blockDim.x + Batch];


    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
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

    for(int i=0;i<4;i++){
        outreg[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        outreg[i][1] = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        outreg[i][2] = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        outreg[i][3] = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    if(relu_pooling == 0){
        int a = tX*4*blockDim.x + tY*4*oside*blockDim.x + chn*oside*oside*blockDim.x + Batch;
        for(int i=0;i<4;i++){
            pOutputs[a + i*oside*deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + 2*deep] = outreg[i][0];
            pOutputs[a + i*oside*deep + 3*deep] = outreg[i][0];
        }
    }else{
        int a = tX*2*blockDim.x + tY*2*oside*blockDim.x/2 + chn*oside*oside*blockDim.x/4 + Batch;
        //relu
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                if(outreg[i][j]<=0)
                    outreg[i][j]=0;
            }
        }

        //max pooling
        rp_reg[0][0] = max(outreg[0][0],max(outreg[0][1],max(outreg[1][0],outreg[1][1])));
        rp_reg[0][1] = max(outreg[0][2],max(outreg[0][3],max(outreg[1][2],outreg[1][3])));
        rp_reg[1][0] = max(outreg[2][0],max(outreg[2][1],max(outreg[3][0],outreg[3][1])));
        rp_reg[1][1] = max(outreg[2][2],max(outreg[2][3],max(outreg[3][2],outreg[3][3])));

        for(int i=0;i<2;i++){
            pOutputs[a + i*oside/2*deep] = rp_reg[i][0];
            pOutputs[a + i*oside/2*deep + deep] = rp_reg[i][1];
        }
    }
}



__global__ void Winofuse4433(int num, int inside, int chn, float* pInputs, float* pFilters, float* pOutputs){
    //data layout is CHWN

    //blockx use to locate the position in a graph
    int tx = blockIdx.x % num;
    int ty = blockIdx.x / num;

    int tN_Input = blockIdx.y;//cal 16N per block      input 
    int tN_Filter = blockIdx.z;//cal 16N per block     Filter

    int tidx = threadIdx.x;//16
    int tidy = threadIdx.y;//16

    //locate input and output
    pInputs = &pInputs[16*gridDim.y*4*tx + ty*4*inside*16*gridDim.y + tN_Input*16];
    pFilters = &pFilters[tN_Filter*16];
    pOutputs = &pOutputs[tidx + tidy*(inside-2)*(inside-2)*16*gridDim.y + 16*gridDim.y*4*tx + ty*4*(inside-2)*16*gridDim.y + tN_Input*16 + tN_Filter*16*(inside-2)*(inside-2)*16*gridDim.y];

    __shared__ float InputCache[36][8][16];
    __shared__ float FilterCache[36][8][16];

    float regCache[6][6] = {{0}};//use to hold input graph and gemm output
    float regCache_n[6][6] = {{0}};
    float regCache_o[6][6] = {{0}};
    float regout[4][4] = {{0}};//use to hold the output

    float A_reg[36] = {0};
    float B_reg[36] = {0};

    for(int sm=0;sm<chn;sm+=8){
        //load filter and Input
        if(tidy<8){
            #pragma unroll
            for(int i=0;i<36;i++){
                FilterCache[i][tidy][tidx] = pFilters[i*chn*chn + sm*chn + tidx + tidy*chn];
            }
        }else{
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache[j][i] = pInputs[i*16*gridDim.y + j*inside*16*gridDim.y + sm*inside*inside*16*gridDim.y + tidx + (tidy-8)*inside*inside*16*gridDim.y];
                }
            }

            float Atd[6][6] = {{0}};

            for(int i=0;i<6;i++){
                Atd[0][i] = 4*regCache[0][i] - 5*regCache[2][i] + regCache[4][i];
                Atd[1][i] = -4*regCache[1][i] -4*regCache[2][i] + regCache[3][i] + regCache[4][i];
                Atd[2][i] = 4*regCache[1][i] -4*regCache[2][i] - regCache[3][i] + regCache[4][i];
                Atd[3][i] = -2*regCache[1][i] - regCache[2][i] + 2*regCache[3][i] + regCache[4][i];
                Atd[4][i] = 2*regCache[1][i] - regCache[2][i] - 2*regCache[3][i] + regCache[4][i];
                Atd[5][i] = 4*regCache[1][i] - 5*regCache[3][i] + regCache[5][i];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                regCache_n[i][0] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
                regCache_n[i][1] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
                regCache_n[i][2] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
                regCache_n[i][3] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][4] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][5] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
            }

            //after transform load the input to shared
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    InputCache[6*j+i][tidy-8][tidx] = regCache_n[j][i];
                }
            }
        }
        
        __syncthreads();

        for(int K=0;K<8;K++){
            //load reg
            #pragma unroll
            for(int i=0;i<36;i++){
                A_reg[i] = FilterCache[i][K][tidy];
                B_reg[i] = InputCache[i][K][tidx];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache_o[j][i] += B_reg[j*6+i]*A_reg[j*6+i];
                }
            }
        }

        __syncthreads();
    }

    //load the outgemm trans to regout
    float Atd[4][6] = {{0}};

    for(int i=0;i<6;i++){
        Atd[0][i] = regCache_o[0][i] + regCache_o[1][i] + regCache_o[2][i] + regCache_o[3][i] + regCache_o[4][i];
        Atd[1][i] = regCache_o[1][i] - regCache_o[2][i] + 2*regCache_o[3][i] - 2*regCache_o[4][i];
        Atd[2][i] = regCache_o[1][i] + regCache_o[2][i] + 4*regCache_o[3][i] + 4*regCache_o[4][i];
        Atd[3][i] = regCache_o[1][i] - regCache_o[2][i] + 8*regCache_o[3][i] - 8*regCache_o[4][i] + regCache_o[5][i];
    }

    for(int i=0;i<4;i++){
        regout[i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
        regout[i][1]  = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
        regout[i][2]  = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
        regout[i][3]  = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
    }

    //output
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            pOutputs[i*gridDim.y*16 + j*(inside-2)*gridDim.y*16] = regout[j][i];
        }
    }
}

__global__ void Winofuse4433_v2(int num, int inside, int chn, float* pInputs, float* pFilters, float* pOutputs){
    //data layout is CHWN

    //blockx use to locate the position in a graph
    int tx = blockIdx.x % num;
    int ty = blockIdx.x / num;

    int tN_Input = blockIdx.y;//cal 32N per block      input 
    int tN_Filter = blockIdx.z;//cal 32N per block     Filter

    int tidx = threadIdx.x;//32
    int tidy = threadIdx.y;//8

    //locate input and output
    pInputs = &pInputs[32*gridDim.y*4*tx + ty*4*inside*32*gridDim.y + tN_Input*32];
    pFilters = &pFilters[tN_Filter*32];
    pOutputs = &pOutputs[tidx + 4*tidy*(inside-2)*(inside-2)*32*gridDim.y + 32*gridDim.y*4*tx + ty*4*(inside-2)*32*gridDim.y + tN_Input*32 + tN_Filter*32*(inside-2)*(inside-2)*32*gridDim.y];

    __shared__ float InputCache[36][4][32];
    __shared__ float FilterCache[36][4][32];

    float regCache[6][6] = {{0}};//use to hold input graph and gemm output
    float regCache_n[6][6] = {{0}};
    float regCache_o[4][6][6] = {{{0}}};
    float regout[4][4][4] = {{{0}}};//use to hold the output

    float A_reg[4][36] = {{0}};
    float B_reg[36] = {0};

    for(int sm=0;sm<chn;sm+=4){
        //load filter and Input
        if(tidy<4){
            #pragma unroll
            for(int i=0;i<36;i++){
                FilterCache[i][tidy][tidx] = pFilters[i*chn*chn + sm*chn + tidx + tidy*chn];
            }
        }else{
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache[j][i] = pInputs[i*32*gridDim.y + j*inside*32*gridDim.y + sm*inside*inside*32*gridDim.y + tidx + (tidy-4)*inside*inside*32*gridDim.y];
                }
            }

            float Atd[6][6] = {{0}};
            
            #pragma unroll
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*regCache[0][i] - 5*regCache[2][i] + regCache[4][i];
                Atd[1][i] = -4*regCache[1][i] -4*regCache[2][i] + regCache[3][i] + regCache[4][i];
                Atd[2][i] = 4*regCache[1][i] -4*regCache[2][i] - regCache[3][i] + regCache[4][i];
                Atd[3][i] = -2*regCache[1][i] - regCache[2][i] + 2*regCache[3][i] + regCache[4][i];
                Atd[4][i] = 2*regCache[1][i] - regCache[2][i] - 2*regCache[3][i] + regCache[4][i];
                Atd[5][i] = 4*regCache[1][i] - 5*regCache[3][i] + regCache[5][i];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                regCache_n[i][0] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
                regCache_n[i][1] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
                regCache_n[i][2] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
                regCache_n[i][3] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][4] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][5] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
            }

            //after transform load the input to shared
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    InputCache[6*j+i][tidy-4][tidx] = regCache_n[j][i];
                }
            }
        }
        
        __syncthreads();

        #pragma unroll
        for(int K=0;K<4;K++){
            //load reg
            #pragma unroll
            for(int i=0;i<36;i++){
                A_reg[0][i] = FilterCache[i][K][4*tidy];
                A_reg[1][i] = FilterCache[i][K][4*tidy+1];
                A_reg[2][i] = FilterCache[i][K][4*tidy+2];
                A_reg[3][i] = FilterCache[i][K][4*tidy+3];
                B_reg[i] = InputCache[i][K][tidx];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache_o[0][j][i] += B_reg[j*6+i]*A_reg[0][j*6+i];
                    regCache_o[1][j][i] += B_reg[j*6+i]*A_reg[1][j*6+i];
                    regCache_o[2][j][i] += B_reg[j*6+i]*A_reg[2][j*6+i];
                    regCache_o[3][j][i] += B_reg[j*6+i]*A_reg[3][j*6+i];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int roll=0;roll<4;roll++){
        //load the outgemm trans to regout
        float Atd[4][6] = {{0}};

        for(int i=0;i<6;i++){
            Atd[0][i] = regCache_o[roll][0][i] + regCache_o[roll][1][i] + regCache_o[roll][2][i] + regCache_o[roll][3][i] + regCache_o[roll][4][i];
            Atd[1][i] = regCache_o[roll][1][i] - regCache_o[roll][2][i] + 2*regCache_o[roll][3][i] - 2*regCache_o[roll][4][i];
            Atd[2][i] = regCache_o[roll][1][i] + regCache_o[roll][2][i] + 4*regCache_o[roll][3][i] + 4*regCache_o[roll][4][i];
            Atd[3][i] = regCache_o[roll][1][i] - regCache_o[roll][2][i] + 8*regCache_o[roll][3][i] - 8*regCache_o[roll][4][i] + regCache_o[roll][5][i];
        }

        for(int i=0;i<4;i++){
            regout[roll][i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
            regout[roll][i][1]  = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
            regout[roll][i][2]  = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
            regout[roll][i][3]  = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        }

    }

    //output
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 0*(inside-2)*(inside-2)*32*gridDim.y] = regout[0][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 1*(inside-2)*(inside-2)*32*gridDim.y] = regout[1][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 2*(inside-2)*(inside-2)*32*gridDim.y] = regout[2][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 3*(inside-2)*(inside-2)*32*gridDim.y] = regout[3][j][i];
        }
    }

    __syncthreads();
}

__global__ void Winofuse4433_v3(int num, int inside, int chn, float* pInputs, float* pFilters, float* pOutputs){
    //data layout is CHWN

    //blockx use to locate the position in a graph
    int tx = blockIdx.x % num;
    int ty = blockIdx.x / num;

    int tN_Input = blockIdx.y;//cal 16N per block      input 
    int tN_Filter = blockIdx.z;//cal 64N per block     Filter

    int tidx = threadIdx.x;//16
    int tidy = threadIdx.y;//16

    //locate input and output
    pInputs = &pInputs[16*gridDim.y*4*tx + ty*4*inside*16*gridDim.y + tN_Input*16];
    pFilters = &pFilters[tN_Filter*64];
    pOutputs = &pOutputs[tidx + 4*tidy*(inside-2)*(inside-2)*16*gridDim.y + 16*gridDim.y*4*tx + ty*4*(inside-2)*16*gridDim.y + tN_Input*16 + tN_Filter*64*(inside-2)*(inside-2)*16*gridDim.y];

    __shared__ float InputCache[36][4][16];
    __shared__ float FilterCache[36][4][64];

    float regCache[6][6] = {{0}};//use to hold input graph and gemm output
    float regCache_n[6][6] = {{0}};
    float regCache_o[4][6][6] = {{{0}}};
    float regout[4][4][4] = {{{0}}};//use to hold the output

    float A_reg[4][36] = {{0}};
    float B_reg[36] = {0};

    for(int sm=0;sm<chn;sm+=4){
        //load filter and Input
        //load filter
        if(tidy<12){
            // for(int i=0;i<3;i++){
            //     #pragma unroll
            //     for(int j=0;j<4;j++){
            //         *((float4 *)(FilterCache[i*12 + tidy][j] + 4*tidx)) = *((float4 *)(pFilters + 4*tidx + j*chn + (i*12+tidy)*chn*chn + sm*chn));
            //     }
            // }
            for(int i=0;i<12;i++){
                *((float4 *)(FilterCache[i*3 + tidy/4][tidy%4] + 4*tidx)) = *((float4 *)(pFilters + 4*tidx + (tidy%4)*chn + (i*3 + tidy/4)*chn*chn + sm*chn));
            }
            
        }else{
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache[j][i] = pInputs[i*16*gridDim.y + j*inside*16*gridDim.y + sm*inside*inside*16*gridDim.y + tidx + (tidy-12)*inside*inside*16*gridDim.y];
                }
            }

            float Atd[6][6] = {{0}};
            
            #pragma unroll
            for(int i=0;i<6;i++){
                Atd[0][i] = 4*regCache[0][i] - 5*regCache[2][i] + regCache[4][i];
                Atd[1][i] = -4*regCache[1][i] -4*regCache[2][i] + regCache[3][i] + regCache[4][i];
                Atd[2][i] = 4*regCache[1][i] -4*regCache[2][i] - regCache[3][i] + regCache[4][i];
                Atd[3][i] = -2*regCache[1][i] - regCache[2][i] + 2*regCache[3][i] + regCache[4][i];
                Atd[4][i] = 2*regCache[1][i] - regCache[2][i] - 2*regCache[3][i] + regCache[4][i];
                Atd[5][i] = 4*regCache[1][i] - 5*regCache[3][i] + regCache[5][i];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                regCache_n[i][0] = 4*Atd[i][0] - 5*Atd[i][2] + Atd[i][4];
                regCache_n[i][1] = -4*Atd[i][1] - 4*Atd[i][2] + Atd[i][3] + Atd[i][4];
                regCache_n[i][2] = 4*Atd[i][1] - 4*Atd[i][2] - Atd[i][3] + Atd[i][4];
                regCache_n[i][3] = -2*Atd[i][1] - Atd[i][2] + 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][4] = 2*Atd[i][1] - Atd[i][2] - 2*Atd[i][3] + Atd[i][4];
                regCache_n[i][5] = 4*Atd[i][1] - 5*Atd[i][3] + Atd[i][5];
            }

            //after transform load the input to shared
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    InputCache[6*j+i][tidy-12][tidx] = regCache_n[j][i];
                }
            }
        }
        
        __syncthreads();

        #pragma unroll
        for(int K=0;K<4;K++){
            //load reg
            #pragma unroll
            for(int i=0;i<36;i++){
                A_reg[0][i] = FilterCache[i][K][4*tidy];
                A_reg[1][i] = FilterCache[i][K][4*tidy+1];
                A_reg[2][i] = FilterCache[i][K][4*tidy+2];
                A_reg[3][i] = FilterCache[i][K][4*tidy+3];
                B_reg[i] = InputCache[i][K][tidx];
            }

            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    regCache_o[0][j][i] += B_reg[j*6+i]*A_reg[0][j*6+i];
                    regCache_o[1][j][i] += B_reg[j*6+i]*A_reg[1][j*6+i];
                    regCache_o[2][j][i] += B_reg[j*6+i]*A_reg[2][j*6+i];
                    regCache_o[3][j][i] += B_reg[j*6+i]*A_reg[3][j*6+i];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int roll=0;roll<4;roll++){
        //load the outgemm trans to regout
        float Atd[4][6] = {{0}};

        for(int i=0;i<6;i++){
            Atd[0][i] = regCache_o[roll][0][i] + regCache_o[roll][1][i] + regCache_o[roll][2][i] + regCache_o[roll][3][i] + regCache_o[roll][4][i];
            Atd[1][i] = regCache_o[roll][1][i] - regCache_o[roll][2][i] + 2*regCache_o[roll][3][i] - 2*regCache_o[roll][4][i];
            Atd[2][i] = regCache_o[roll][1][i] + regCache_o[roll][2][i] + 4*regCache_o[roll][3][i] + 4*regCache_o[roll][4][i];
            Atd[3][i] = regCache_o[roll][1][i] - regCache_o[roll][2][i] + 8*regCache_o[roll][3][i] - 8*regCache_o[roll][4][i] + regCache_o[roll][5][i];
        }

        for(int i=0;i<4;i++){
            regout[roll][i][0] = Atd[i][0] + Atd[i][1] + Atd[i][2] + Atd[i][3] + Atd[i][4];
            regout[roll][i][1]  = Atd[i][1] - Atd[i][2] + 2*Atd[i][3] - 2*Atd[i][4];
            regout[roll][i][2]  = Atd[i][1] + Atd[i][2] + 4*Atd[i][3] + 4*Atd[i][4];
            regout[roll][i][3]  = Atd[i][1] - Atd[i][2] + 8*Atd[i][3] - 8*Atd[i][4] + Atd[i][5];
        }

    }

    //output
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            pOutputs[i*gridDim.y*16 + j*(inside-2)*gridDim.y*16 + 0*(inside-2)*(inside-2)*16*gridDim.y] = regout[0][j][i];
            pOutputs[i*gridDim.y*16 + j*(inside-2)*gridDim.y*16 + 1*(inside-2)*(inside-2)*16*gridDim.y] = regout[1][j][i];
            pOutputs[i*gridDim.y*16 + j*(inside-2)*gridDim.y*16 + 2*(inside-2)*(inside-2)*16*gridDim.y] = regout[2][j][i];
            pOutputs[i*gridDim.y*16 + j*(inside-2)*gridDim.y*16 + 3*(inside-2)*(inside-2)*16*gridDim.y] = regout[3][j][i];
        }
    }

    __syncthreads();
}

__global__ void Conv_fuse(int num, int inside, int chn, float* pInputs, float* pFilters, float* pOutputs){
    //data layout is CHWN

    //blockx use to locate the position in a graph
    int tx = blockIdx.x % num;
    int ty = blockIdx.x / num;

    int tN_Input = blockIdx.y;//cal 32N per block      input 
    int tN_Filter = blockIdx.z;//cal 32N per block     Filter

    int tidx = threadIdx.x;//32
    int tidy = threadIdx.y;//8

    //locate input and output
    pInputs = &pInputs[32*gridDim.y*4*tx + ty*4*inside*32*gridDim.y + tN_Input*32];
    pFilters = &pFilters[tN_Filter*32];
    pOutputs = &pOutputs[tidx + 4*tidy*(inside-2)*(inside-2)*32*gridDim.y + 32*gridDim.y*4*tx + ty*4*(inside-2)*32*gridDim.y + tN_Input*32 + tN_Filter*32*(inside-2)*(inside-2)*32*gridDim.y];

    __shared__ float InputCache[36][8][32];
    __shared__ float FilterCache[9][8][32];

    float Inreg[6][6] = {{0}};
    float Fireg[4][3][3] = {{{0}}};
    float F_out[4][4][4] = {{{0}}};//use to hold the output

    // float A_reg[4][36] = {{0}};
    // float B_reg[36] = {0};

    for(int sm=0;sm<chn;sm+=8){
        //load filter and Input
        if(tidy<4){
            #pragma unroll
            for(int i=0;i<9;i++){
                FilterCache[i][tidy][tidx] = pFilters[i*chn + sm*9*chn + tidx + tidy*9*chn];
                FilterCache[i][tidy+4][tidx] = pFilters[i*chn + sm*9*chn + tidx + (tidy+4)*9*chn];
                // FilterCache[i][tidy][tidx + 32] = pFilters[i*chn + sm*9*chn + tidx + tidy*9*chn + 32];
            }
        }else{
            #pragma unroll
            for(int i=0;i<6;i++){
                #pragma unroll
                for(int j=0;j<6;j++){
                    InputCache[i + j*6][tidy-4][tidx] = pInputs[i*32*gridDim.y + j*inside*32*gridDim.y + sm*inside*inside*32*gridDim.y + tidx + (tidy-4)*inside*inside*32*gridDim.y];
                    InputCache[i + j*6][tidy][tidx] = pInputs[i*32*gridDim.y + j*inside*32*gridDim.y + sm*inside*inside*32*gridDim.y + tidx + (tidy)*inside*inside*32*gridDim.y];
                }
            }
        }
        
        __syncthreads();

        #pragma unroll
        for(int K=0;K<8;K++){
            //load graph
            #pragma unroll
            for(int i=0;i<36;i++){
                Inreg[i/6][i%6] = InputCache[i][K][tidx];
            }

            //load filter
            #pragma unroll
            for(int i=0;i<4;i++){
                #pragma unroll
                for(int j=0;j<9;j++){
                    Fireg[i][j/3][j%3] = FilterCache[j][K][4*tidy + i];
                }
            }

            //cal output
            #pragma unroll
            for(int i=0;i<4;i++){
                //loc1
                F_out[i][0][0] += (Inreg[0][0]*Fireg[i][0][0] + Inreg[0][1]*Fireg[i][0][1] + Inreg[0][2]*Fireg[i][0][2] + Inreg[1][0]*Fireg[i][1][0] + Inreg[1][1]*Fireg[i][1][1] + Inreg[1][2]*Fireg[i][1][2] + Inreg[2][0]*Fireg[i][2][0] + Inreg[2][1]*Fireg[i][2][1] + Inreg[2][2]*Fireg[i][2][2]);
                //loc2
                F_out[i][0][1] += (Inreg[0][1]*Fireg[i][0][0] + Inreg[0][2]*Fireg[i][0][1] + Inreg[0][3]*Fireg[i][0][2] + Inreg[1][1]*Fireg[i][1][0] + Inreg[1][2]*Fireg[i][1][1] + Inreg[1][3]*Fireg[i][1][2] + Inreg[2][1]*Fireg[i][2][0] + Inreg[2][2]*Fireg[i][2][1] + Inreg[2][3]*Fireg[i][2][2]);
                //loc3
                F_out[i][0][2] += (Inreg[0][2]*Fireg[i][0][0] + Inreg[0][3]*Fireg[i][0][1] + Inreg[0][4]*Fireg[i][0][2] + Inreg[1][2]*Fireg[i][1][0] + Inreg[1][3]*Fireg[i][1][1] + Inreg[1][4]*Fireg[i][1][2] + Inreg[2][2]*Fireg[i][2][0] + Inreg[2][3]*Fireg[i][2][1] + Inreg[2][4]*Fireg[i][2][2]);
                //loc4
                F_out[i][0][3] += (Inreg[0][3]*Fireg[i][0][0] + Inreg[0][4]*Fireg[i][0][1] + Inreg[0][5]*Fireg[i][0][2] + Inreg[1][3]*Fireg[i][1][0] + Inreg[1][4]*Fireg[i][1][1] + Inreg[1][5]*Fireg[i][1][2] + Inreg[2][3]*Fireg[i][2][0] + Inreg[2][4]*Fireg[i][2][1] + Inreg[2][5]*Fireg[i][2][2]);
                /////////////////
                //loc1
                F_out[i][1][0] += (Inreg[1][0]*Fireg[i][0][0] + Inreg[1][1]*Fireg[i][0][1] + Inreg[1][2]*Fireg[i][0][2] + Inreg[2][0]*Fireg[i][1][0] + Inreg[2][1]*Fireg[i][1][1] + Inreg[2][2]*Fireg[i][1][2] + Inreg[3][0]*Fireg[i][2][0] + Inreg[3][1]*Fireg[i][2][1] + Inreg[3][2]*Fireg[i][2][2]);
                //loc2
                F_out[i][1][1] += (Inreg[1][1]*Fireg[i][0][0] + Inreg[1][2]*Fireg[i][0][1] + Inreg[1][3]*Fireg[i][0][2] + Inreg[2][1]*Fireg[i][1][0] + Inreg[2][2]*Fireg[i][1][1] + Inreg[2][3]*Fireg[i][1][2] + Inreg[3][1]*Fireg[i][2][0] + Inreg[3][2]*Fireg[i][2][1] + Inreg[3][3]*Fireg[i][2][2]);
                //loc3
                F_out[i][1][2] += (Inreg[1][2]*Fireg[i][0][0] + Inreg[1][3]*Fireg[i][0][1] + Inreg[1][4]*Fireg[i][0][2] + Inreg[2][2]*Fireg[i][1][0] + Inreg[2][3]*Fireg[i][1][1] + Inreg[2][4]*Fireg[i][1][2] + Inreg[3][2]*Fireg[i][2][0] + Inreg[3][3]*Fireg[i][2][1] + Inreg[3][4]*Fireg[i][2][2]);
                //loc4
                F_out[i][1][3] += (Inreg[1][3]*Fireg[i][0][0] + Inreg[1][4]*Fireg[i][0][1] + Inreg[1][5]*Fireg[i][0][2] + Inreg[2][3]*Fireg[i][1][0] + Inreg[2][4]*Fireg[i][1][1] + Inreg[2][5]*Fireg[i][1][2] + Inreg[3][3]*Fireg[i][2][0] + Inreg[3][4]*Fireg[i][2][1] + Inreg[3][5]*Fireg[i][2][2]);
                ////////////////
                //loc1
                F_out[i][2][0] += (Inreg[2][0]*Fireg[i][0][0] + Inreg[2][1]*Fireg[i][0][1] + Inreg[2][2]*Fireg[i][0][2] + Inreg[3][0]*Fireg[i][1][0] + Inreg[3][1]*Fireg[i][1][1] + Inreg[3][2]*Fireg[i][1][2] + Inreg[4][0]*Fireg[i][2][0] + Inreg[4][1]*Fireg[i][2][1] + Inreg[4][2]*Fireg[i][2][2]);
                //loc2
                F_out[i][2][1] += (Inreg[2][1]*Fireg[i][0][0] + Inreg[2][2]*Fireg[i][0][1] + Inreg[2][3]*Fireg[i][0][2] + Inreg[3][1]*Fireg[i][1][0] + Inreg[3][2]*Fireg[i][1][1] + Inreg[3][3]*Fireg[i][1][2] + Inreg[4][1]*Fireg[i][2][0] + Inreg[4][2]*Fireg[i][2][1] + Inreg[4][3]*Fireg[i][2][2]);
                //loc3
                F_out[i][2][2] += (Inreg[2][2]*Fireg[i][0][0] + Inreg[2][3]*Fireg[i][0][1] + Inreg[2][4]*Fireg[i][0][2] + Inreg[3][2]*Fireg[i][1][0] + Inreg[3][3]*Fireg[i][1][1] + Inreg[3][4]*Fireg[i][1][2] + Inreg[4][2]*Fireg[i][2][0] + Inreg[4][3]*Fireg[i][2][1] + Inreg[4][4]*Fireg[i][2][2]);
                //loc4
                F_out[i][2][3] += (Inreg[2][3]*Fireg[i][0][0] + Inreg[2][4]*Fireg[i][0][1] + Inreg[2][5]*Fireg[i][0][2] + Inreg[3][3]*Fireg[i][1][0] + Inreg[3][4]*Fireg[i][1][1] + Inreg[3][5]*Fireg[i][1][2] + Inreg[4][3]*Fireg[i][2][0] + Inreg[4][4]*Fireg[i][2][1] + Inreg[4][5]*Fireg[i][2][2]);
                ////////////////
                //loc1
                F_out[i][3][0] += (Inreg[3][0]*Fireg[i][0][0] + Inreg[3][1]*Fireg[i][0][1] + Inreg[3][2]*Fireg[i][0][2] + Inreg[4][0]*Fireg[i][1][0] + Inreg[4][1]*Fireg[i][1][1] + Inreg[4][2]*Fireg[i][1][2] + Inreg[5][0]*Fireg[i][2][0] + Inreg[5][1]*Fireg[i][2][1] + Inreg[5][2]*Fireg[i][2][2]);
                //loc2
                F_out[i][3][1] += (Inreg[3][1]*Fireg[i][0][0] + Inreg[3][2]*Fireg[i][0][1] + Inreg[3][3]*Fireg[i][0][2] + Inreg[4][1]*Fireg[i][1][0] + Inreg[4][2]*Fireg[i][1][1] + Inreg[4][3]*Fireg[i][1][2] + Inreg[5][1]*Fireg[i][2][0] + Inreg[5][2]*Fireg[i][2][1] + Inreg[5][3]*Fireg[i][2][2]);
                //loc3
                F_out[i][3][2] += (Inreg[3][2]*Fireg[i][0][0] + Inreg[3][3]*Fireg[i][0][1] + Inreg[3][4]*Fireg[i][0][2] + Inreg[4][2]*Fireg[i][1][0] + Inreg[4][3]*Fireg[i][1][1] + Inreg[4][4]*Fireg[i][1][2] + Inreg[5][2]*Fireg[i][2][0] + Inreg[5][3]*Fireg[i][2][1] + Inreg[5][4]*Fireg[i][2][2]);
                //loc4
                F_out[i][3][3] += (Inreg[3][3]*Fireg[i][0][0] + Inreg[3][4]*Fireg[i][0][1] + Inreg[3][5]*Fireg[i][0][2] + Inreg[4][3]*Fireg[i][1][0] + Inreg[4][4]*Fireg[i][1][1] + Inreg[4][5]*Fireg[i][1][2] + Inreg[5][3]*Fireg[i][2][0] + Inreg[5][4]*Fireg[i][2][1] + Inreg[5][5]*Fireg[i][2][2]);
            }
        }

        __syncthreads();
    }

    //output
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 0*(inside-2)*(inside-2)*32*gridDim.y] = F_out[0][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 1*(inside-2)*(inside-2)*32*gridDim.y] = F_out[1][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 2*(inside-2)*(inside-2)*32*gridDim.y] = F_out[2][j][i];
            pOutputs[i*gridDim.y*32 + j*(inside-2)*gridDim.y*32 + 3*(inside-2)*(inside-2)*32*gridDim.y] = F_out[3][j][i];
        }
    }

    __syncthreads();
}
