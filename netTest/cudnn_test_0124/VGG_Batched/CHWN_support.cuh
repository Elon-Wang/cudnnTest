
// BIAS ADDITION
// version 1
/*
// 简单的CHWN格式bias加法核函数 - 仅支持float类型
__global__ void addBias_CHWN_kernel(
    float* data,       // 输入输出数据 [C,H,W,N]
    const float* bias, // bias数据 [C]
    int C, int H, int W, int N
) {
    int c = blockIdx.x;    // channel维度
    int h = blockIdx.y;    // height维度
    int w = blockIdx.z;    // width维度
    int n = threadIdx.x;   // batch维度
    
    if (c < C && h < H && w < W && n < N) {
        int idx = ((c * H + h) * W + w) * N + n;
        data[idx] += bias[c];
    }
}

// 主机端调用函数
void addBias_CHWN(float* data, const float* bias, int C, int H, int W, int N) {
    dim3 grid(C, H, W);
    dim3 block(N);
    
    addBias_CHWN_kernel<<<grid, block>>>(data, bias, C, H, W, N);
}
*/

/*
//version 2
__global__ void addBias_linear_kernel_optimized(
    float* data,       // 输入输出数据 [C,H,W,N]
    const float* bias, // bias数据 [C]
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    
    // 处理float4对齐的部分
    float4* data4 = reinterpret_cast<float4*>(data);
    int N4 = N / 4;  // 每4个N打包成一个float4
    int total_size_vec4 = C * H * W * N4;
    
    for(int idx4 = tid; idx4 < total_size_vec4; idx4 += stride) {
        // 计算对应的channel索引
        int n4 = idx4 % N4;
        int w = (idx4 / N4) % W;
        int h = (idx4 / (N4 * W)) % H;
        int c = idx4 / (N4 * W * H);
        
        // 读取bias值
        float b = bias[c];
        
        // 读取数据
        float4 val = data4[idx4];
        
        // 加上bias
        val.x += b;
        val.y += b;
        val.z += b;
        val.w += b;
        
        // 写回数据
        data4[idx4] = val;
    }
    
    // 处理剩余的非对齐元素
    int aligned_N = (N / 4) * 4;
    int total_size = C * H * W * N;
    
    for(int idx = tid + C * H * W * aligned_N; idx < total_size; idx += stride) {
        int n = idx % N;
        int w = (idx / N) % W;
        int h = (idx / (N * W)) % H;
        int c = idx / (N * W * H);
        
        data[idx] += bias[c];
    }
}
*/

__global__ void addBias_linear_kernel_optimized(
    float* data,      
    const float* bias,
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    
    int total_size = C * H * W * N;
    
    // 直接按元素处理，让硬件处理内存合并访问
    for(int idx = tid; idx < total_size; idx += stride) {
        int n = idx % N;
        int w = (idx / N) % W;
        int h = (idx / (N * W)) % H;
        int c = idx / (N * W * H);
        
        data[idx] += bias[c];
    }
}

void addBias_CHWN(
    float* data,
    const float* bias,
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int total_size = C * H * W * N;
    
    // 计算block数量
    int num_blocks = min(65535, (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    addBias_linear_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
        data, bias, C, H, W, N);
}



// MAX POOLING
// version 1
/*
// 最大池化的CUDA核函数 - CHWN格式
__global__ void maxPool_CHWN_kernel(
    float* output,      // 输出数据
    const float* input, // 输入数据
    int C, int H, int W, int N,
    int pool_size,      // 池化窗口大小
    int stride          // 步长
) {
    // 计算输出的维度
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    // 获取当前线程处理的位置
    int c = blockIdx.x;    // channel维度
    int h = blockIdx.y;    // height维度
    int w = blockIdx.z;    // width维度
    int n = threadIdx.x;   // batch维度
    
    if (c < C && h < H_out && w < W_out && n < N) {
        // 计算输入和输出的基础索引
        int h_start = h * stride;
        int w_start = w * stride;
        
        // 在池化窗口中找最大值
        float maxVal = -INFINITY;
        
        // 遍历池化窗口
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_start + ph;
                int w_in = w_start + pw;
                
                // CHWN格式的索引计算
                int in_idx = ((c * H + h_in) * W + w_in) * N + n;
                float val = input[in_idx];
                maxVal = max(maxVal, val);
            }
        }
        
        // 写入输出
        int out_idx = ((c * H_out + h) * W_out + w) * N + n;
        output[out_idx] = maxVal;
    }
}

// 主机端调用函数
void maxPool_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N,
    int pool_size = 2,    // 默认2x2池化
    int stride = 2        // 默认步长2
) {
    // 计算输出维度
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    // 设置grid和block维度
    dim3 grid(C, H_out, W_out);
    dim3 block(N);
    
    // 启动核函数
    maxPool_CHWN_kernel<<<grid, block>>>(
        output, input,
        C, H, W, N,
        pool_size, stride
    );
}
*/

// version 2
__global__ void maxPool_linear_kernel_optimized(
    float* output,      
    const float* input, 
    int C, int H, int W, int N,
    int pool_size,
    int stride
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int thread_stride = gridDim.x * BLOCK_SIZE;
    
    // 计算输出维度
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    // 计算总的输出元素数量
    int total_out_size = C * H_out * W_out * N;
    
    // 每个线程处理多个输出位置
    for(int out_idx = tid; out_idx < total_out_size; out_idx += thread_stride) {
        // 反向计算对应的c,h,w,n位置
        int n = out_idx % N;
        int w = (out_idx / N) % W_out;
        int h = (out_idx / (N * W_out)) % H_out;
        int c = out_idx / (N * W_out * H_out);
        
        // 计算输入的起始位置
        int h_start = h * stride;
        int w_start = w * stride;
        
        // 在池化窗口中找最大值
        float maxVal = -INFINITY;
        
        // 遍历池化窗口
        for(int ph = 0; ph < pool_size; ph++) {
            for(int pw = 0; pw < pool_size; pw++) {
                int h_in = h_start + ph;
                int w_in = w_start + pw;
                
                // CHWN格式的输入索引计算
                int in_idx = ((c * H + h_in) * W + w_in) * N + n;
                maxVal = max(maxVal, input[in_idx]);
            }
        }
        
        // 写入输出
        output[out_idx] = maxVal;
    }
}

void maxPool_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N,
    int pool_size = 2,
    int stride = 2
) {
    const int BLOCK_SIZE = 256;
    
    // 计算输出维度
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    int total_out_size = C * H_out * W_out * N;
    
    // 计算block数量
    int num_blocks = min(65535, (total_out_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    maxPool_linear_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
        output, input,
        C, H, W, N,
        pool_size, stride
    );
}


// ACTIVATION FUNCTION
// version 1
/*
// ReLU激活函数的CUDA核函数 - CHWN格式
__global__ void relu_CHWN_kernel(
    float* output,      // 输出数据
    const float* input, // 输入数据
    int C, int H, int W, int N
) {
    // 获取当前线程处理的位置
    int c = blockIdx.x;    // channel维度
    int h = blockIdx.y;    // height维度
    int w = blockIdx.z;    // width维度
    int n = threadIdx.x;   // batch维度
    
    if (c < C && h < H && w < W && n < N) {
        // CHWN格式的索引计算
        int idx = ((c * H + h) * W + w) * N + n;
        // ReLU操作: max(0, x)
        output[idx] = max(0.0f, input[idx]);
    }
}

// 主机端调用函数
void relu_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N
) {
    // 设置grid和block维度
    dim3 grid(C, H, W);
    dim3 block(N);
    
    // 启动核函数
    relu_CHWN_kernel<<<grid, block>>>(output, input, C, H, W, N);
}
*/

//version2
/*
// 使用2D block的ReLU实现
__global__ void relu_CHWN_kernel_2D(
    float* output,      // 输出数据
    const float* input, // 输入数据
    int C, int H, int W, int N
) {
    // 使用2D block: (32, 32)处理一个tile
    const int tx = threadIdx.x;  // 0-31
    const int ty = threadIdx.y;  // 0-31
    
    // 每个block处理32x32的数据
    const int TILE_DIM = 32;
    
    // 计算当前block处理的起始位置
    int c = blockIdx.x;
    int h_start = blockIdx.y * TILE_DIM;
    int w_start = blockIdx.z * TILE_DIM;
    
    // 每个线程处理一个(h,w)位置的所有N个元素
    int h = h_start + ty;
    int w = w_start + tx;
    
    if (c < C && h < H && w < W) {
        // 计算当前位置的起始索引
        int base_idx = ((c * H + h) * W + w) * N;
        
        // 每个线程循环处理N个元素
        #pragma unroll 4
        for (int n = 0; n < N; n++) {
            output[base_idx + n] = max(0.0f, input[base_idx + n]);
        }
    }
}

// 改进的主机端调用函数
void relu_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N
) {
    const int TILE_DIM = 32;
    
    // 设置grid和block维度
    dim3 grid(C, 
              (H + TILE_DIM - 1) / TILE_DIM, 
              (W + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);  // 使用32x32的2D block
    
    // 启动核函数
    relu_CHWN_kernel_2D<<<grid, block>>>(output, input, C, H, W, N);
}
*/

//version 3
/*
__global__ void relu_linear_kernel(
    float* output,      
    const float* input, 
    int total_size     // C*H*W*N
) {
    const int BLOCK_SIZE = 256;
    
    // 计算当前线程的全局索引
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 每个线程处理多个元素，stride更大，确保每个线程有足够的工作量
    for(int idx = tid; idx < total_size; idx += gridDim.x * BLOCK_SIZE) {
        output[idx] = max(0.0f, input[idx]);
    }
}

// 主机端调用函数
void relu_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int total_size = C * H * W * N;
    
    int num_blocks = min(65535, (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    relu_linear_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, total_size);
}
*/

// version 4
/*
__global__ void relu_linear_kernel_vec4(
    float4* output,      
    const float4* input, 
    int total_size_vec4    // total_size/4，表示float4的数量
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 每个线程处理连续的float4数据
    for(int idx = tid; idx < total_size_vec4; idx += gridDim.x * BLOCK_SIZE) {
        float4 val = input[idx];
        
        // 对float4中的每个元素进行ReLU
        val.x = max(0.0f, val.x);
        val.y = max(0.0f, val.y);
        val.z = max(0.0f, val.z);
        val.w = max(0.0f, val.w);
        
        output[idx] = val;
    }
}

// 处理剩余的非4对齐元素
__global__ void relu_linear_kernel_remainder(
    float* output,
    const float* input,
    int total_size,
    int aligned_size
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 处理剩余元素
    for(int idx = aligned_size + tid; idx < total_size; idx += gridDim.x * BLOCK_SIZE) {
        output[idx] = max(0.0f, input[idx]);
    }
}

void relu_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int total_size = C * H * W * N;
    
    // 计算能用float4处理的元素数量
    int aligned_size = (total_size / 4) * 4;
    int total_size_vec4 = aligned_size / 4;
    
    // 计算block数量
    int num_blocks = min(65535, (total_size_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    num_blocks = max(1, num_blocks);
    
    // 处理float4对齐的部分
    if (total_size_vec4 > 0) {
        relu_linear_kernel_vec4<<<num_blocks, BLOCK_SIZE>>>(
            reinterpret_cast<float4*>(output),
            reinterpret_cast<const float4*>(input),
            total_size_vec4
        );
    }
    
    // 处理剩余的元素
    if (aligned_size < total_size) {
        int remainder_blocks = min(65535, ((total_size - aligned_size) + BLOCK_SIZE - 1) / BLOCK_SIZE);
        remainder_blocks = max(1, remainder_blocks);
        
        relu_linear_kernel_remainder<<<remainder_blocks, BLOCK_SIZE>>>(
            output,
            input,
            total_size,
            aligned_size
        );
    }
}
*/

//version 5
__global__ void relu_linear_kernel_optimized(
    float* __restrict__ output,      
    const float* __restrict__ input, 
    int total_size
) {
    const int BLOCK_SIZE = 256;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    
    // 处理float4对齐的部分
    float4* out4 = reinterpret_cast<float4*>(output);
    const float4* in4 = reinterpret_cast<const float4*>(input);
    int total_size_vec4 = total_size / 4;
    
    // 每个线程处理多个float4
    #pragma unroll 2
    for(int idx = tid; idx < total_size_vec4; idx += stride) {
        float4 val = in4[idx];
        val.x = max(0.0f, val.x);
        val.y = max(0.0f, val.y);
        val.z = max(0.0f, val.z);
        val.w = max(0.0f, val.w);
        out4[idx] = val;
    }
    
    // 处理剩余的元素
    int aligned_size = total_size_vec4 * 4;
    for(int idx = aligned_size + tid; idx < total_size; idx += stride) {
        output[idx] = max(0.0f, input[idx]);
    }
}

void relu_CHWN(
    float* output,
    const float* input,
    int C, int H, int W, int N
) {
    const int BLOCK_SIZE = 256;
    int total_size = C * H * W * N;
    
    // 简单地计算block数量，让每个线程至少处理4个元素
    int target_threads = total_size / 4;
    int num_blocks = min(65535, (target_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    num_blocks = max(1, num_blocks);
    
    relu_linear_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
        output, input, total_size);
}