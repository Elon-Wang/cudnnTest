#pragma once
#include <cudnn.h>
#include "error_util.h"

#ifndef WARMUP
#define WARMUP
__global__ void warmup(){}
#endif

void wrapedConv_CuDNN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output){

    float one = 1.0, zero = 0.0;
    int size;
    cudnnStatus_t status;
    cudnnHandle_t handle;
    
    cudnnTensorDescriptor_t xdesc, ydesc;
    cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
    
    cudnnConvolutionDescriptor_t conv_desc;
    // cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

    float *extra;

    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");
    status = cudnnCreateTensorDescriptor(&xdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
    status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat4Conv, chn, inside, inside);//input
    if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
    status = cudnnCreateTensorDescriptor(&ydesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
    int oside =inside +2*padding -2;
    status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat4Conv, numOfFilter, oside, oside);//output
    if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
    status = cudnnCreateFilterDescriptor(&wdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
    status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, numOfFilter, chn, 3, 3);//filter
    if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
    status = cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
    if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");
    status = cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
    // algo = (cudnnConvolutionFwdAlgo_t)7; //CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    status = cudnnGetConvolutionForwardWorkspaceSize(handle,
        xdesc,
        wdesc,
        conv_desc,
        ydesc,
        algo,
        (size_t *)&(size)
    );
    cudaMalloc((void **) &extra, size);


    status = cudnnConvolutionForward(handle, &one, xdesc, m1, wdesc, m2, conv_desc, algo, extra, size, &zero, ydesc, *output);
    if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed:%s\n",cudnnGetErrorString(status));

    cudaDeviceSynchronize();
    cudnnDestroy(handle);
    cudaFree(extra);
};