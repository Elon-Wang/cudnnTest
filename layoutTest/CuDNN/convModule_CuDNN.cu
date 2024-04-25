#include "cudnn.h"
#include <cuda_runtime.h>
#include <stdio.h>
// #include "../include/para.c"
#include "util.h"
#include <string.h>

int main(int argc, char** argv){
    cudaSetDevice(0);
    /*----------input file-------------*/
    // printf("Implicit_gemm:0 Implicit_precomp_gemm:1 GEMM:2\n");
    // printf("Winograd:6 Wino_nonfused:7\n");
    const char dirPath[] = "/home/wangq/project/cudnnTest/layoutTest/CuDNN";
    char inputName[100];
    char filterName[100];
    strcpy(inputName, dirPath);
    strcpy(filterName, dirPath);
    strcat(inputName, "/data/input.bin");
    strcat(filterName, "/data/filter.bin");

    /*----------parameters & data setup-------------*/
    // int bat4Conv = 64;
    // int inside = 22;
    // int chn =128;
    // int numOfFilter = 128;
    // int padding = 1;
    int bat4Conv = atoi(argv[1]);
    int inside = atoi(argv[2]);
    int chn = atoi(argv[3]);
    int numOfFilter = atoi(argv[4]);
    cudnnConvolutionFwdAlgo_t model = (argc ==5)? CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: (cudnnConvolutionFwdAlgo_t) atoi(argv[5]);
    int padding = 1;

    int oside = inside + 2*padding-2;
    int nInput = bat4Conv*inside*inside*chn;
    int nFilter = numOfFilter*3*3*chn;
    int nOutput = bat4Conv*oside*oside*numOfFilter;

    float *input_cpu  = get_parameter(inputName , nInput);
    float *filter_cpu = get_parameter(filterName, nFilter);
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
    status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat4Conv, chn, inside, inside);//input
    if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
    status = cudnnCreateTensorDescriptor(&ydesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
    status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat4Conv, numOfFilter, oside, oside);//output
    if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
    status = cudnnCreateFilterDescriptor(&wdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
    status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, numOfFilter, chn, 3, 3);//filter
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
    double FLOP = ((3*3*chn)*chn + chn)*1.0e-9*inside*inside*bat4Conv;
    double tflops = (FLOP/avetime);

    // printf("Batch:%d Inside:%d chn:%d\n",Batch,inside,chn);
    printf("time:%lf us TFLOPS:%lf\n",(avetime)*1000,tflops);
    
    /*  4. Copy back and free  */
     char outputName[100];
    strcpy(outputName, dirPath);
    strcat(outputName, "/data/ConvModule_CuDNN.bin");
    // printf("outputName path:%s\n",outputName);
    int cnt = save_parameter(outputName, nOutput, output_cpu);
    
    cudnnDestroy(handle);
    cudaFree(extra);
    cudaFree(input_gpu);
    cudaFree(output_gpu);
    cudaFree(filter_gpu);
    
    return 0;
}
