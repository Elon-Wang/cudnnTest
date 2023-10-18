#include "cudnn.h"
#include <cuda_runtime.h>
#include "testCase.h"
#include "Kernel.h"


class ConvTest{
    public:
    int bat4Conv;
    int inside, inside_beta;
    int chn, numOfFilter;
    int padding =1;
    int blockn;
    int M, N, K;
    bool MCheck, NCheck, KCheck;
    int MSize, NSize,KSize;
    int blockx, blocky;
    int bat4Gemm = 36;
    int oside;

    int nInputTran, nKernelTran, nGemmOutput, nOutput;

    // float *inputTran_cpu;
    float *inputTran_gpu;
    float *kernelTran_gpu;
    float *gemmOutput_gpu;
    float *output_gpu;
    float *output_cpu;

    char outFileName[40] = "default_Name.bin";
    char fileLastName[20] = "default_LastName";

    float singleTime =0;
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid= false;

    virtual bool testValid(testCase tc);
    float testPerformance(testCase tc);
    void reportPerformance(testCase tc);
    virtual void execut(testCase tc)=0;
    ConvTest(){

    }
};

bool ConvTest::testValid(testCase tc){
    if(true){ // TODO: complete the Validity test;
        inside = tc.side;
        chn = tc.chn;
        numOfFilter = tc.numOfFilter;
        bat4Conv = tc.bat4Conv;

        int marginOfInputSide = (inside+2*padding-6)%4;
        bool sideCheck = ( marginOfInputSide == 0 )? true: false ;
        inside_beta = sideCheck? inside +2*padding : ( inside + 2*padding + 4 - marginOfInputSide);
        oside = inside + 2* padding -2;
        blockn = (inside_beta -2) /4;

        M = bat4Conv * blockn * blockn;
        N = numOfFilter;
        K = chn;
        MCheck = ( M % 128 == 0 )? true: false;
        NCheck = ( N % 128 == 0 )? true: false;
        KCheck = ( K % 8 == 0 )? true: false;

        MSize = MCheck? M: ((M/128 +1) * 128);
        NSize = NCheck? N: ((N/128 +1) * 128);
        KSize = KCheck? K: ((K/8 +1) * 8);

        blockx = (MSize+127)/128;
        blocky = (NSize+127)/128;

        nInputTran = 36 * MSize * KSize;
        nKernelTran = 36 * NSize * KSize;
        nGemmOutput = 36 * MSize * NSize;
        nOutput = numOfFilter * oside *oside *chn;

        cudaMalloc((void **) &inputTran_gpu, nInputTran<<2); 
        cudaMalloc((void **) &kernelTran_gpu, nKernelTran<<2); 
        cudaMalloc((void **) &gemmOutput_gpu, nGemmOutput<<2); 
        cudaMalloc((void **) &output_gpu, nOutput<<2);        
        output_cpu = (float*)malloc(nOutput *sizeof(float));



        return true;
    } else{
        printf("**************test case invalid*************\n");
        return false;
    }
}

float ConvTest::testPerformance(testCase tc) {

    warmup<<<1,1>>>();
    int cnt =10;
    float timeSeries[cnt];
    avgDelay = 0;
    char tcidx[5];
    sprintf(tcidx,"%d",tc.index);
    strcat(outFileName, tcidx);
    strcat(outFileName, fileLastName);

    for(int i =0; i < cnt;i++) {
        cudaEvent_t start1,stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);

        cudaEventRecord(start1, NULL);
        execut(tc);
        cudaEventRecord(stop1, NULL);

        cudaEventSynchronize(start1);
        cudaEventSynchronize(stop1);

        cudaEventElapsedTime(&singleTime, start1, stop1);
        
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);

        timeSeries[i] = singleTime;
        avgDelay += singleTime;
    }
    
    avgDelay /= cnt;
    minDelay = timeSeries[0];
    maxDelay = timeSeries[0];
    for (int i=1; i<cnt; i++){
        maxDelay = (maxDelay > timeSeries[i])? maxDelay: timeSeries[i];
        minDelay = (minDelay < timeSeries[i])? minDelay: timeSeries[i];
    }

    cudaMemcpy(output_cpu, output_gpu, nOutput<<2, cudaMemcpyDeviceToHost);
    // printf("first element:%f\n",kernelTran_cpu[0]);
    save_parameter(outFileName, nOutput, output_cpu);
    return avgDelay;
}

void ConvTest::reportPerformance(testCase tc){
    printf("testCase:%d\t avgDelay:%f\tminDelay:%f\tmaxDelay:%f\n",tc.index, avgDelay, minDelay,maxDelay);
}

class new0: public ConvTest{
    public:
    new0(){
        strcpy(fileLastName,"/Conv_new0.bin");
        strcpy(outFileName, "./data/tc");
    }
    virtual void execut(testCase tc){
        wino_input_trans_nchw_suitFor128<<<dim3( chn,blockn,blockn),dim3( bat4Conv,1,1)>>>(inside, inside_beta, MSize, KSize, padding, tc.input_gpu, inputTran_gpu);
        wino_kernel_trans_nchw_suitFor128<<<dim3( chn,1,1),dim3(tc.numOfFilter,1,1)>>>(NSize, KSize, tc.kernel_gpu, kernelTran_gpu);
        GEMM_batch_256_128x128_KMKN<<<dim3(blockx, blocky, bat4Gemm), dim3(256,1,1)>>> (MSize,NSize,KSize,1, inputTran_gpu, kernelTran_gpu,0, gemmOutput_gpu);
        wino_invers_nchw_suitFor128<<<dim3(bat4Conv,blockn,blockn), dim3(numOfFilter,1,1)>>>(oside, MSize, NSize, gemmOutput_gpu, output_gpu);
    }
};

class cudnnConv: public ConvTest{
    public:
    float one = 1.0, zero = 0.0;
    int size;
    cudnnStatus_t status;
    cudnnHandle_t handle;
    
    cudnnTensorDescriptor_t xdesc, ydesc;
    cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
    
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    float *extra;
    float avetime = 0;

    virtual bool testValid(testCase tc);
    cudnnConv(){
        strcpy(fileLastName,"/Conv_new0.bin");
        strcpy(outFileName, "./data/tc");
    }
    virtual void execut(testCase tc){
        status = cudnnConvolutionForward(handle, &one, xdesc, tc.input_gpu, wdesc, tc.kernel_gpu, conv_desc, algo, extra, size, &zero, ydesc, output_gpu);
        if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed:%s\n",cudnnGetErrorString(status));
    }
    ~cudnnConv(){
        cudnnDestroy(handle);
        cudaFree(extra);
        // cudaFree(input_gpu);
        // cudaFree(output_gpu);
        // cudaFree(filter_gpu);
    }
};

bool cudnnConv::testValid(testCase tc){
    if(true){
        inside = tc.side;
        chn = tc.chn;
        numOfFilter = tc.numOfFilter;
        bat4Conv = tc.bat4Conv;
        oside = inside +2*padding -2;

        nOutput = numOfFilter * oside *oside * chn;
        cudaMalloc((void **) &output_gpu, nOutput<<2);
        cudaMemset((void *) output_gpu, 0, nOutput<<2);
        output_cpu = (float*) malloc(nOutput*4);

        status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");
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
        status = cudnnCreateConvolutionDescriptor(&conv_desc);
        if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
        status = cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
        if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");
        status = cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH);
        if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
        algo = (cudnnConvolutionFwdAlgo_t)7; //CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        status = cudnnGetConvolutionForwardWorkspaceSize(handle,
            xdesc,
            wdesc,
            conv_desc,
            ydesc,
            algo,
            (size_t *)&(size)
        );
        cudaMalloc((void **) &extra, size);
        return true;
    } else {
        printf("**************test case invalid*************\n");
        return false;
    }
}
