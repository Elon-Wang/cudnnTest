#pragma once
#include <bits/stdc++.h>
#include "util.h"
#include "DataLayoutTrans.cuh"

template <class value_type>
struct Layer_t
{
    int inputs;     // input channels
    int outputs;    // output channels
    DataLayout layout_in, layout_out;

    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;

    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0)
    {}

    Layer_t(int _inputs, int _outputs, int _kernel_dim, DataLayout _layout_in, DataLayout _layout_out, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim), layout_in(_layout_in), layout_out(_layout_out)
    {
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
            // weights_path = std::string("vggData/") + std::string(fname_weights);
            // bias_path = std::string("vggData/") + std::string(fname_bias);
        }
        else
        {
            weights_path = std::string("vggData/") + fname_weights; 
            bias_path = std::string("vggData/") + fname_bias;
        }
        // printf("weights_path: %s\n", weights_path.c_str());
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }

    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }

    private:

    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};


// demonstrate different ways of setting tensor descriptor
#define SIMPLE_TENSOR_DESCRIPTOR
// #define ND_TENSOR_DESCRIPTOR

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
    if (tensorFormat == -1){
        printf("undefined Tensor Format\n");
        return;
    }
#if defined(SIMPLE_TENSOR_DESCRIPTOR)
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}