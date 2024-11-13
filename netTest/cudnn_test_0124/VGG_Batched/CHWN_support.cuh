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