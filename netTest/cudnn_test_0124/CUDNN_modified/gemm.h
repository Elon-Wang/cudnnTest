// TODO: 
// **** take care of the position of the pixel in different data_layout. *****
// maybe need to reposition of the data point using a densing function before full connection.
// consider the most simple case of the nchw layout.
// or maybe for the different data layout, the network need to be trained again and using onther set of weights. 
// only need to take care if batch is 1 or not. channel information is lost in densing layers.
// take care of either batched or not.
// so using the gemm for multiple batch, and gemv for only single batch.
// take care of either trans or not.
// need to figure out the n, c, h, w corresponding to which

#if !defined(_GEMM_H_)
#define _GEMM_H_

#include <cuda.h> // CUDA_VERSION
#include <cublas_v2.h>
#include "error_util.h"

void gemm (cublasHandle_t cublasHandle, int batch, int m, int k, 
            double alpha, 
            const double *weight, const double *src_data,
            double beta, double *dst_data)
{
    cublasDgemm(
                cublasHandle,
                CUBLAS_OP_T,   //矩阵A的属性参数，不转置，按列优先
                CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
                m,             //矩阵B^T、C^T的行数
                batch,         //矩阵A^T、C^T的列数
                k,             //B^T的列数，A^T的行数，此处也可为A_COL,一样的
                &alpha,        //alpha的值
                weight,        //左矩阵，为B^T
                k,             //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
                src_data,      //右矩阵，为A^T
                k,             //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
                &beta,         //beta的值
                dst_data,      //结果矩阵C
                m              //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );
        // 小心这里一堆的leading dimension 和 什么 m, n, k.
}

void gemm (cublasHandle_t cublasHandle, int batch, int m, int k, 
            float alpha, 
            const float *weight, const float *src_data, 
            float beta, float *dst_data)
{
    cublasSgemm(
                cublasHandle,
                CUBLAS_OP_T,   //矩阵A的属性参数，不转置，按列优先
                CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
                m,             //矩阵B^T、C^T的行数
                batch,         //矩阵A^T、C^T的列数
                k,             //B^T的列数，A^T的行数，此处也可为A_COL,一样的
                &alpha,        //alpha的值
                weight,        //左矩阵，为B^T
                k,             //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
                src_data,      //右矩阵，为A^T
                k,             //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
                &beta,         //beta的值
                dst_data,      //结果矩阵C
                m              //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );
        // 小心这里一堆的leading dimension 和 什么 m, n, k.
}

#endif //_GEMM_H_