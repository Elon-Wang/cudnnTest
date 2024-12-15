#include <bits/stdc++.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include "error_util.h"
#include "util.h"

#include "DataLayoutTrans.cuh"
// #include "wrapedConv_NCHW.cuh"
// #include "wrapedConv_NHWC.cuh"
// #include "wrapedConv_CHWN.cuh"
#include "wrapedConv_CuDNN.cuh"

cudnnHandle_t handle;
cudnnTensorDescriptor_t xdesc, ydesc;
cudnnFilterDescriptor_t wdesc;
cudnnConvolutionDescriptor_t conv_desc;


void readAllocInit(const char* fname, int size, float** data_h, float** data_d)
    {
        readAllocMemcpy(fname, size, data_h, data_d);
    }

void BatchTesting(int opt, int bat4Conv, int inside, int chn, int numOfFilter, int padding, const char* fileName,const char* inputName, float *inputTran_gpu, float *filterTran_gpu, float *gemmOutput_gpu);

int main(int argc, char *argv[]){
    // cudaSetDevice(1);
    cudaDeviceReset();
    // config
    // char filtername[100];

    int device =1;
    cudaSetDevice(device);
    int n = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    int opt = getCmdLineArgumentInt(argc, (const char **)argv, "opt");
    int layer = getCmdLineArgumentInt(argc, (const char **)argv, "layer"); // 1: layer1, 2: layer2, 3: layer3, 4: layer4, 5: layer5, 6: layer6-7, 7: layer8, 8: layer9-10, 9: layer11-13

    int h, c, numOfFilter;
    char filtername[100];
    switch (layer)
    {
    case 1:
        h = 224; c = 3; numOfFilter = 64;
        sprintf(filtername, "./vggData/conv1.bin");
        break;
    case 2:
        h = 224; c = 64; numOfFilter = 64;
        sprintf(filtername, "./vggData/conv2.bin");
        break;
    case 3:
        h = 112; c = 64; numOfFilter = 128;
        sprintf(filtername, "./vggData/conv3.bin");
        break;
    case 4:
        h = 112; c = 128; numOfFilter = 128;
        sprintf(filtername, "./vggData/conv4.bin");
        break;
    case 5:
        h = 56; c = 128; numOfFilter = 256;
        sprintf(filtername, "./vggData/conv5.bin");
        break;
    case 6:
        h = 56; c = 256; numOfFilter = 256;
        sprintf(filtername, "./vggData/conv7.bin");
        break;
    case 7:
        h = 28; c = 256; numOfFilter = 512;
        sprintf(filtername, "./vggData/conv8.bin");
        break;
    case 8:
        h = 28; c = 512; numOfFilter = 512;
        sprintf(filtername, "./vggData/conv10.bin");
        break;
    case 9:
        h = 14; c = 512; numOfFilter = 512;
        sprintf(filtername, "./vggData/conv13.bin");
        break;
    default:
        printf("The layer is not right\n");
        break;
    }

    
    // sprintf(filtername, "./vggData/conv%d.bin", layer);
    // int h = 224; int c = 64; int numOfFilter = 64;
    // char filtername2[100] = "./vggData/conv2.bin";

    // int h = 112; int c = 64; int numOfFilter = 128;
    // char filtername3[100] = "./vggData/conv3.bin";
    // int h = 112; int c = 128;int numOfFilter = 128;
    // char filtername4[100] = "./vggData/conv4.bin";

    // int h = 56; int c = 128; int numOfFilter = 256;
    // char filtername5[100] = "./vggData/conv5.bin";
    // int h = 56; int c = 256; int numOfFilter = 256;
    // char filtername6[100] = "./vggData/conv7.bin";

    // int h = 28; int c = 256; int numOfFilter = 512;
    // char filtername7[100] = "./vggData/conv8.bin";
    // int h = 28; int c = 512; int numOfFilter = 512;
    // char filtername[100] = "./vggData/conv10.bin";

    // int h = 14; int c = 512; int numOfFilter = 512;
    // char filtername9[100] = "./vggData/conv13.bin";
    cudnnStatus_t status;
    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");
    status = cudnnCreateTensorDescriptor(&xdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
    status = cudnnCreateTensorDescriptor(&ydesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
    status = cudnnCreateFilterDescriptor(&wdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");

    printf("opt:%d\n",opt);

    float *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    long long nInputTran = 36*3200*64 *n;
    long long nFilterTran = 36*512*512 *n;
    long long nGemmOutput = 36*3200*128 *n;
    checkCudaErrors( cudaMalloc(&inputTran_gpu,  nInputTran<<2) );
    checkCudaErrors( cudaMalloc(&filterTran_gpu,  nFilterTran<<2) );
    checkCudaErrors( cudaMalloc(&gemmOutput_gpu,  nGemmOutput<<2) );
    warmup<<<1,1>>>();

    for (int i=0;i<20;i++){
        char file_name[40];
        sprintf(file_name, "./binImage/file_%d.bin", i); //20 different file

        BatchTesting(opt, n, h, c, numOfFilter, 1, filtername, file_name, inputTran_gpu, filterTran_gpu, gemmOutput_gpu);
    }
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyTensorDescriptor(ydesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    return 0;
}

void BatchTesting(int opt, int bat4Conv, int side, int chn, int numOfFilter, int padding, const char * filterName, const char* inputName, float *inputTran_gpu, float *filterTran_gpu, float *gemmOutput_gpu){
    int nFilter = 9 * numOfFilter * chn;
    int nInput = side*side *chn *bat4Conv;

    // construct the working space
    float *filterData_h = get_parameter(filterName, nFilter);
    // float *filterData_h = NULL;
    float *filterData_d = NULL; 

    float *srcData = NULL, *dstData = NULL;
    // float *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // float* imgData_h = (float*)malloc(side*side*bat4Conv *chn * sizeof(float));

    checkCudaErrors( cudaMalloc(&filterData_d, nFilter * sizeof(float)  ) );

    checkCudaErrors( cudaMalloc(&srcData, side * side * sizeof(float) * bat4Conv * chn ) );
    checkCudaErrors( cudaMalloc(&dstData, side * side * sizeof(float) * bat4Conv * numOfFilter) );
    
    // checkCudaErrors( cudaMalloc(&dstData, 224 * 224 * sizeof(float) * bat4Conv * 64) );
    checkCudaErrors( cudaMemcpy(filterData_d, filterData_h,
                                    sizeof(float) * nFilter,
                                    cudaMemcpyHostToDevice) );

    // read File
    float *imgData_h = get_parameter(inputName, nInput);

    checkCudaErrors( cudaMemcpy(srcData, imgData_h, nInput *sizeof(float), cudaMemcpyHostToDevice) );


    float avetime = 0;
    cudaEvent_t start1,stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, NULL); 

    if(opt ==0) {
        printf("Conv_NCHW\n");
        // wrapedConv_NCHW(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
        layoutManager(DataLayout::NCHW, DataLayout::NCHW, bat4Conv, side, chn, numOfFilter, padding, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, &dstData);
    } else if (opt ==1) {
        printf("Conv_NHWC\n");
        // wrapedConv_NHWC(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
        layoutManager(DataLayout::NHWC, DataLayout::NHWC, bat4Conv, side, chn, numOfFilter, padding, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, &dstData);
        // wrapedConv_NCHW(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu , &dstData);
    } else if (opt ==2) {
        printf("Conv_CHWN\n");
        // wrapedConv_CHWN(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
        layoutManager(DataLayout::CHWN, DataLayout::CHWN, bat4Conv, side, chn, numOfFilter, padding, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu, &dstData);
        // wrapedConv_NCHW_3(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu , &dstData);
    } else if (opt ==3 || opt ==4 || opt ==5 || opt ==6) {
        printf("Conv_CuDNN\n");
        wrapedConv_CuDNN(bat4Conv, side, chn, numOfFilter, padding, srcData, filterData_d, &dstData, opt, handle, xdesc, ydesc, wdesc, conv_desc);
    } else{
        printf("The option is not right\n");
    }
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop1, NULL);
    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&avetime, start1, stop1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);


    int nConvOutput = bat4Conv * side * side * numOfFilter;
    float *convOutput_cpu = (float *)malloc(nConvOutput * sizeof(float));
    cudaMemcpy(convOutput_cpu, dstData, nConvOutput<<2, cudaMemcpyDeviceToHost);
    // float *convOutput_cpu = (float *)malloc(nInput * sizeof(float));
    // cudaMemcpy(convOutput_cpu, dstData, nInput<<2, cudaMemcpyDeviceToHost);
    
    // print the result.
    printf("time:%lf ms\n", (avetime));


    printf("convOutput_cpu[0]:%lf\n",convOutput_cpu[0]);
    // for (int i=0;i<5;i++){
    //     printf("%lf ",convOutput_cpu[i]);
    // }printf("\n");
    if ( convOutput_cpu[0] == 0.0f) {
        //printf("Test3:\n 1:%f\t 2:%f\n",output_cpu[0], output_cpu[1]);
        printf("moduleConv error\n");
    }
    free(convOutput_cpu);
    free(imgData_h);
    cudaFree(srcData);
    cudaFree(dstData);

}
