#include "cudnn.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/para.c"
#include <string.h>

int main(int argc, char** argv){
    cudaSetDevice(0);
    /*----------input file-------------*/
    // printf("Implicit_gemm:0 Implicit_precomp_gemm:1 GEMM:2\n");
    // printf("Winograd:6 Wino_nonfused:7\n");

    const char inputname[] = "../data/input.bin";
    const char filtername[] = "../data/filter.bin";
    const char outputName[] = "../data/Cu_output2.bin";

    /*----------parameters & data setup-------------*/
    int model = atoi(argv[1]);
    int Batch = atoi(argv[2]);
    int inside = atoi(argv[3]);
    int chn = atoi(argv[4]);
    int padding = 1;
    int oside = inside + 2*padding-2;

    int nInput = Batch*inside*inside*chn;
    int nFilter = chn*3*3*chn;
    int nOutput = Batch*oside*oside*chn;

    float *input_cpu  = get_parameter(inputname , nInput);
    float *filter_cpu = get_parameter(filtername, nFilter);
    float *output_cpu = (float*) malloc(nOutput*4);

    float *input_gpu,*filter_gpu,*output_gpu;

    cudaMalloc((void **) &input_gpu , nInput<<2);
    cudaMalloc((void **) &filter_gpu, nFilter<<2);
    cudaMalloc((void **) &output_gpu, nOutput<<2);

    cudaMemcpy(input_gpu , input_cpu , nInput <<2, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, filter_cpu, nFilter<<2, cudaMemcpyHostToDevice);
    cudaMemset((void *) output_gpu, 0, nOutput<<2);


    /*  2. cuDNN preparation  */
    cudnnStatus_t status;
    float one = 1.0, zero = 0.0;
    int size;

    cudnnHandle_t handle;
    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

    cudnnTensorDescriptor_t xdesc, ydesc;
    cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
    status = cudnnCreateTensorDescriptor(&xdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
    status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, Batch, chn, inside, inside);//input
    if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
    status = cudnnCreateTensorDescriptor(&ydesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
    status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, Batch, chn, oside, oside);//output
    if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
    status = cudnnCreateFilterDescriptor(&wdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
    status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, chn, chn, 3, 3);//filter
    if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
    cudnnConvolutionDescriptor_t conv_desc;
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
    status = cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
    if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");
    status = cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");

    // GEMM algorithm 
    cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)model;

    status = cudnnGetConvolutionForwardWorkspaceSize(handle,
       xdesc,
       wdesc,
       conv_desc,
       ydesc,
       algo,
       (size_t *)&(size)
    );

    float *extra;
    cudaMalloc((void **) &extra, size);

    float avetime = 0;
    /*  3. Computing  */

    //warm up
    status = cudnnConvolutionForward(handle, &one, xdesc, input_gpu, wdesc, filter_gpu, conv_desc, algo, extra, size, &zero, ydesc, output_gpu);

    cudaEvent_t start1,stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, NULL);  
    for(int x=0;x<10;x++){
        status = cudnnConvolutionForward(handle, &one, xdesc, input_gpu, wdesc, filter_gpu, conv_desc, algo, extra, size, &zero, ydesc, output_gpu);
    }
    cudaEventRecord(stop1, NULL);
    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&avetime, start1, stop1);

    if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed:%s\n",cudnnGetErrorString(status));
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    
    cudaMemcpy(output_cpu, output_gpu, nOutput<<2, cudaMemcpyDeviceToHost);
    
    avetime = avetime/10;
    double FLOP = ((3*3*chn)*chn + chn)*1.0e-9*inside*inside*Batch;
    double tflops = (FLOP/avetime);

    // printf("Batch:%d Inside:%d chn:%d\n",Batch,inside,chn);
    printf("time:%lf us TFLOPS:%lf\n",(avetime)*1000,tflops);
    
    /*  4. Copy back and free  */
    int cnt = save_parameter(outputName, nOutput, output_cpu);
    
    cudnnDestroy(handle);
    cudaFree(extra);
    cudaFree(input_gpu);
    cudaFree(output_gpu);
    cudaFree(filter_gpu);
    
    return 0;
}
