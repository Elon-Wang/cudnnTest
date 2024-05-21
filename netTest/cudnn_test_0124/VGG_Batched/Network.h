#pragma once
#include <bits/stdc++.h>
#include "Layer.h"
#include "matrixOp.h"
#include "wrapedConv_NCHW.cuh"
// #include "wrapedConv_NHWC.cuh"

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnMathType_t mathType = CUDNN_FMA_MATH;
    cudnnHandle_t cudnnHandle;
    // cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cudnnDropoutDescriptor_t dropoutDesc;
    cublasHandle_t cublasHandle;
    float *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;

    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );
        checkCUDNN( cudnnCreateDropoutDescriptor(&dropoutDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }

    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyDropoutDescriptor(dropoutDesc) );
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }

    public:

    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;

    network_t()
    {
        convAlgorithm = -1;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        // tensorFormat = CUDNN_TENSOR_NCHW;
        tensorFormat = CUDNN_TENSOR_NHWC;
        createHandles();    
    };

    ~network_t()
    {
        destroyHandles();
    }

    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }

    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }

    void setTensorFormat(const cudnnTensorFormat_t& format)
    {
        tensorFormat = format;
    }

    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }

    void fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        // resize(n*dim_y, dstData);

        scaling_type alpha = scaling_type(1), beta = scaling_type(1);

        for(int iter =0;iter <n;iter++) {
            checkCudaErrors( cudaMemcpy(*dstData + iter*dim_y, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        }
        
        if (n ==1) {
            // place bias into dstData        
            gemv(cublasHandle, dim_x, dim_y, alpha,
                    ip.data_d, srcData, beta,*dstData);
        } else {
            gemm(cublasHandle, n, dim_y, dim_x, alpha, 
                    ip.data_d, srcData, beta, *dstData);
        }

        h = 1; w = 1; c = dim_y;
    }

    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA) );
 
        const int convDims = 2;
        int padA[convDims] = {1,1};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;

        // Math are done in FP32 when tensor are in FP16.
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT;
        }

        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );

        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            printf("incorrect convolution algorithm\n");
            exit(-1);
            // algo = results[0].algo;            

            // int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT; 
            // int returnedAlgoCount = -1;
            // cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

            // // Choose the best according to the preference
            // std::cout << "Testing cudnnGetConvolutionForwardAlgorithm_v7 ...\n";
            // checkCUDNN( cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
            //                                                   srcTensorDesc,
            //                                                   filterDesc,
            //                                                   convDesc,
            //                                                   dstTensorDesc,
            //                                                   requestedAlgoCount,
            //                                                   &returnedAlgoCount,
            //                                                   results));
            // for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
            //     printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
            //         cudnnGetErrorString(results[algoIndex].status), 
            //         results[algoIndex].algo, results[algoIndex].time, 
            //         (unsigned long long)results[algoIndex].memory);
            // }

            // // New way of finding the fastest config
            // // Setup for findFastest call
            // std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            // checkCUDNN( cudnnFindConvolutionForwardAlgorithm(cudnnHandle, 
            //                                                 srcTensorDesc,
            //                                                 filterDesc,
            //                                                 convDesc,
            //                                                 dstTensorDesc,
            //                                                 requestedAlgoCount,
            //                                                 &returnedAlgoCount,
            //                                                 results));
            // for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
            //     printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
            //         cudnnGetErrorString(results[algoIndex].status), 
            //         results[algoIndex].algo, results[algoIndex].time, 
            //         (unsigned long long)results[algoIndex].memory);
            // }
            
            // algo = results[0].algo;

        } else {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }
        
        cudnnSetConvolutionMathType( convDesc, mathType);

        // resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        // resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }

    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        // resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }

    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        // resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }

    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        // resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

    // void dropoutForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData, const value_type p){
        
    //     checkCUDNN( cudnnSetDropoutDescriptor( dropoutDesc,
    //                                             cudnnHandle,
    //                                             p,

    //                                             ))

    // }

    void convMethodChoose(const Layer_t<value_type>& conv, int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData, bool choice){
        if (choice) {
            // cudaEvent_t ts1, ts2, ts3;
            // cudaEventCreate(&ts1);
            // cudaEventCreate(&ts2);
            // cudaEventCreate(&ts3);

            // cudaEventRecord(ts1, NULL);
            wrapedConv_NCHW(n, h, c, conv.outputs, 1, srcData, conv.data_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu , dstData);
            // wrapedConv_NCHW(n, h, c, conv.outputs, 1, srcData, conv.data_d, dstData);
            // wrapedConv_NHWC(n, h, c, conv.outputs, 1, srcData, conv.data_d,  inputTran_gpu, filterTran_gpu, gemmOutput_gpu, dstData);

            cudaDeviceSynchronize();
            // cudaEventRecord(ts2, NULL);
            c = conv.outputs;
            setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

            printf("addBias count\n");
            addBias(dstTensorDesc, conv, c, *dstData);

            // cudaEventRecord(ts3, NULL);
            // cudaEventSynchronize(ts1);
            // cudaEventSynchronize(ts2);
            // cudaEventSynchronize(ts3);
            // float timeCache;
            // cudaEventElapsedTime(&timeCache, ts1, ts2);
            // printf("ts2:%lf ms\n", (timeCache));
            // cudaEventElapsedTime(&timeCache, ts2, ts3);
            // printf("ts3:%lf ms\n", (timeCache));
        } else{
            convoluteForward(conv, n, c, h, w, srcData, dstData);
        }
    }

    std::vector<int> classify_example_modified(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& conv3,
                          const Layer_t<value_type>& conv4,
                          const Layer_t<value_type>& conv5,
                          const Layer_t<value_type>& conv6,
                          const Layer_t<value_type>& conv7,
                          const Layer_t<value_type>& conv8,
                          const Layer_t<value_type>& conv9,
                          const Layer_t<value_type>& conv10,
                          const Layer_t<value_type>& conv11,
                          const Layer_t<value_type>& conv12,
                          const Layer_t<value_type>& conv13,
                          const Layer_t<value_type>& fc14,
                          const Layer_t<value_type>& fc15,
                          const Layer_t<value_type>& fc16,
                          const int batch, const int chn, const int side, bool testChoice)
    {

        // TBD
        // changing the file reading functions;

        int n,c,h,w;
        h = w = side; n = batch; c = chn;

        value_type *srcData = NULL, *dstData = NULL;
        value_type* imgData_h = (value_type*)malloc(side*side*batch *chn * sizeof(value_type));

        readImage(fname, imgData_h, n*c*h*w);

        // printf("srcData first setup, valid:%d, %d\n",(srcData!=NULL),srcData);
        // std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, side * side * sizeof(value_type) * batch * 64) );
        checkCudaErrors( cudaMalloc(&dstData, side * side * sizeof(value_type) * batch * 64) );
        
        int nInputTran = 36*3200*64 *batch;
        int nFilterTran = 36*512*512 *batch;
        int nGemmOutput = 36*3200*128 *batch;
        cudaMalloc((void **) &inputTran_gpu,  nInputTran<<2);
        cudaMalloc((void **) &filterTran_gpu, nFilterTran<<2);
        cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2);
        // checkCudaErrors( cudaMalloc(&inputTran_gpu, 36* 3200 * 64) );
        // checkCudaErrors( cudaMalloc(&filterTran_gpu, 36* 512 * 512) );
        // checkCudaErrors( cudaMalloc(&gemmOutput_gpu, 36* 3200 * 128) );

        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    side * side *sizeof(value_type) * batch * chn,
                                    cudaMemcpyHostToDevice) );

        // code used for debug
        // int cnt =0;
        // std::cout<<cnt++<<std::endl;
        // printf("n:%d c:%d h:%d w:%d\n",n,c,h,w);

        // int nConvOutput = n * side * side * conv1.outputs;
        // // float *convOutput_gpu;
        // cudaMalloc((void **) &dstData, nConvOutput* sizeof(value_type));
        // wrapedConv_NCHW(n, h, c, conv1.outputs, 1, srcData, conv1.data_d, &dstData);
        // convoluteForward(conv1, n, c, h, w, srcData, &dstData);

        // int nOutput = n*c*h*w;
        // printf("n:%d c:%d h:%d w:%d nOutput:%d \n",n,c,h,w,nOutput);
        // float *output_cpu = (float*) malloc(nOutput*4);
        // cudaMemcpy(output_cpu, dstData, nOutput<<2, cudaMemcpyDeviceToHost);
        // // save_parameter(outputName, nOutput, output_cpu);

        // for(int i=0;i<50;i++) {
        //     printf("%f ", output_cpu[i]);
        // }printf("\n");

        float avetime =0;
        // cudaEvent_t start1,stop1,ts1,ts2,ts3,ts4,ts5,ts6;
        cudaEvent_t start1,stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);

        cudaEventRecord(start1, NULL);

        convMethodChoose(conv1, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
        convMethodChoose(conv2, n, c, h, w, srcData, &dstData, testChoice);      
        activationForward(n, c, h, w, dstData, &srcData);
		poolForward(n, c, h, w, srcData, &dstData);
        
        convMethodChoose(conv3, n, c, h, w, dstData, &srcData, testChoice);
        activationForward(n, c, h, w, srcData, &dstData);
        convMethodChoose(conv4, n, c, h, w, dstData, &srcData, testChoice);
        activationForward(n, c, h, w, srcData, &dstData);
		poolForward(n, c, h, w, dstData, &srcData);

        convMethodChoose(conv5, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
        convMethodChoose(conv6, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
        convMethodChoose(conv7, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
		poolForward(n, c, h, w, srcData, &dstData);

        convMethodChoose(conv8, n, c, h, w, dstData, &srcData, testChoice);
        activationForward(n, c, h, w, srcData, &dstData);
        convMethodChoose(conv9, n, c, h, w, dstData, &srcData, testChoice);
        activationForward(n, c, h, w, srcData, &dstData);
        convMethodChoose(conv10, n, c, h, w, dstData, &srcData, testChoice);
        activationForward(n, c, h, w, srcData, &dstData);
		poolForward(n, c, h, w, dstData, &srcData);

        convMethodChoose(conv11, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
        convMethodChoose(conv12, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
        convMethodChoose(conv13, n, c, h, w, srcData, &dstData, testChoice);
        activationForward(n, c, h, w, dstData, &srcData);
		poolForward(n, c, h, w, srcData, &dstData);

        fullyConnectedForward(fc14, n, c, h, w, dstData, &srcData);
        activationForward(n, c, h, w, srcData, &dstData);
        fullyConnectedForward(fc15, n, c, h, w, dstData, &srcData);       
        activationForward(n, c, h, w, srcData, &dstData);
        fullyConnectedForward(fc16, n, c, h, w, dstData, &srcData);
        softmaxForward(n, c, h, w, srcData, &dstData);

        cudaEventRecord(stop1, NULL);

        //cuDNN and cuBLAS library calls are asynchronous w.r.t. the host.
        // Need a device sync here before copying back the results.
        checkCudaErrors (cudaDeviceSynchronize());
        
        const int max_digits = 1000;

        // Take care of half precision
        value_type result[n][max_digits];
        cudaMemcpy(result, dstData, n*max_digits<<2, cudaMemcpyDeviceToHost);
        
        // not captible with multiple batch inference
        std::vector<int> ret;
        int id = 0;
        for (int batch =0; batch <n; batch++) 
        {
            for (int i = 1; i < max_digits; i++)
            {
                if (Convert<scaling_type>(result[batch][id]) < Convert<scaling_type>(result[batch][i])) {
                    id = i;
                }
            }
            // std::cout << "Batch "<< batch <<" Resulting weights from Softmax:" << id << std::endl;
            ret.push_back(id);
        }
        
        // float timeCache;
        // cudaEventElapsedTime(&timeCache, start1, ts1);
        // printf("ts1:%lf ms\n", (timeCache));
        // cudaEventElapsedTime(&timeCache, ts1, ts2);
        // printf("ts2:%lf ms\n", (timeCache));
        // cudaEventElapsedTime(&timeCache, ts2, ts3);
        // printf("ts3:%lf ms\n", (timeCache));
        // cudaEventElapsedTime(&timeCache, ts3, ts4);
        // printf("ts4:%lf ms\n", (timeCache));
        // cudaEventElapsedTime(&timeCache, ts4, ts5);
        // printf("ts5:%lf ms\n", (timeCache));
        // cudaEventElapsedTime(&timeCache, ts5, ts6);
        // printf("ts6:%lf ms\n", (timeCache));

        cudaEventElapsedTime(&avetime, start1, stop1);
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        printf("time:%lf ms\n", (avetime/n));

        bool debug = false;
        if (debug){
            printDeviceVector(n, c*h*w, dstData);
        }

        checkCudaErrors( cudaFree(inputTran_gpu));
        checkCudaErrors( cudaFree(filterTran_gpu));
        checkCudaErrors( cudaFree(gemmOutput_gpu));

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return ret;
    }
};