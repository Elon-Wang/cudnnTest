#pragma once
#include <cudnn.h>
#include "error_util.h"

#ifndef WARMUP
#define WARMUP
__global__ void warmup(){}
#endif

void wrapedConv_CuDNN(int bat4Conv, int inside, int& chn, int numOfFilter, int padding, float *m1, float *m2, float ** output, int mode, cudnnHandle_t &handle, cudnnTensorDescriptor_t &xdesc, cudnnTensorDescriptor_t &ydesc, cudnnFilterDescriptor_t &wdesc, cudnnConvolutionDescriptor_t &conv_desc){

    float one = 1.0, zero = 0.0;
    long long size;
    cudnnStatus_t status;
    // cudnnHandle_t handle;
    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
    
    // cudnnTensorDescriptor_t xdesc, ydesc;
    // cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
    
    // cudnnConvolutionDescriptor_t conv_desc;
    // Nonfused timing error, batch =1 is slower than batch=2,4,8,16
    // cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

    switch (mode)
    {
    case 3:
        tensorFormat = CUDNN_TENSOR_NCHW;
        algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        break;
    case 4:
        tensorFormat = CUDNN_TENSOR_NCHW;
        algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    case 5:
        tensorFormat = CUDNN_TENSOR_NHWC;
        algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        break;
    case 6:
        tensorFormat = CUDNN_TENSOR_NHWC;
        algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    default:
        printf("The mode is not right\n");
        break;
    }

    float *extra;

    status = cudnnSetTensor4dDescriptor(xdesc, tensorFormat, CUDNN_DATA_FLOAT, bat4Conv, chn, inside, inside);//input
    if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
    int oside =inside +2*padding -2;
    status = cudnnSetTensor4dDescriptor(ydesc, tensorFormat, CUDNN_DATA_FLOAT, bat4Conv, numOfFilter, oside, oside);//output
    if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
    status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, numOfFilter, chn, 3, 3);//filter
    if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
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
    
    cudaFree(extra);
    
};