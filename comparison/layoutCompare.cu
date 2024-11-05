#include <bits/stdc++.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include "error_util.h"
#include "util.h"

#include "wrapedConv_NCHW.cuh"
#include "wrapedConv_NHWC.cuh"
#include "wrapedConv_CHWN.cuh"
#include "wrapedConv_CuDNN.cuh"



void readAllocInit(const char* fname, int size, float** data_h, float** data_d)
    {
        readAllocMemcpy(fname, size, data_h, data_d);
    }

void BatchTesting(int opt, int bat4Conv, int inside, int chn, int numOfFilter, int padding, const char* fileName,const char* inputName);

int main(int argc, char *argv[]){
    cudaSetDevice(1);
    cudaDeviceReset();
    // config
    // char filtername[100];

    // int h = 224; int c = 3; int numOfFilter = 64;
    // char filtername[100] = "./vggData/conv1.bin";
    // int h = 224; int c = 64; int numOfFilter = 64;
    // char filtername[100] = "./vggData/conv2.bin";

    // int h = 112; int c = 64; int numOfFilter = 128;
    // char filtername[100] = "./vggData/conv3.bin";
    // int h = 112; int c = 128;int numOfFilter = 128;
    // char filtername[100] = "./vggData/conv4.bin";

    // int h = 56; int c = 128; int numOfFilter = 256;
    // char filtername[100] = "./vggData/conv5.bin";
    // int h = 56; int c = 256; int numOfFilter = 256;
    // char filtername[100] = "./vggData/conv7.bin";

    // int h = 28; int c = 256; int numOfFilter = 512;
    // char filtername[100] = "./vggData/conv8.bin";
    // int h = 28; int c = 512; int numOfFilter = 512;
    // char filtername[100] = "./vggData/conv10.bin";

    int h = 14; int c = 512; int numOfFilter = 512;
    char filtername[100] = "./vggData/conv13.bin";


    int n = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    int opt = getCmdLineArgumentInt(argc, (const char **)argv, "opt");

    printf("opt:%d\n",opt);

    warmup<<<1,1>>>();

    for (int i=0;i<10;i++){
        char file_name[40];
        sprintf(file_name, "./binImage/file_%d.bin", i); //20 different file

        BatchTesting(opt, n, h, c, numOfFilter, 1, filtername, file_name);
    }

    return 0;
}

void BatchTesting(int opt, int bat4Conv, int side, int chn, int numOfFilter, int padding, const char * filterName, const char* inputName){
    int nFilter = 9 * numOfFilter * chn;
    int nInput = side*side *chn *bat4Conv;

    // construct the working space
    float *filterData_h = get_parameter(filterName, nFilter);
    // float *filterData_h = NULL;
    float *filterData_d = NULL; 

    float *srcData = NULL, *dstData = NULL;
    float *inputTran_gpu, *filterTran_gpu, *gemmOutput_gpu;
    // float* imgData_h = (float*)malloc(side*side*bat4Conv *chn * sizeof(float));

    checkCudaErrors( cudaMalloc(&filterData_d, nFilter * sizeof(float)  ) );

    checkCudaErrors( cudaMalloc(&srcData, 224 * 224 * sizeof(float) * bat4Conv * 64 ) );
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
        printf("Conv_NCHW_4\n");
        wrapedConv_NCHW_4(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
    } else if (opt ==1) {
        printf("Conv_NCHW is deprecated\n");
        return;
        // wrapedConv_NCHW(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu , &dstData);
    } else if (opt ==2) {
        printf("Conv_NCHW_3 is deprecated\n");
        return;
        // wrapedConv_NCHW_3(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, inputTran_gpu, filterTran_gpu, gemmOutput_gpu , &dstData);
    } else if (opt ==3) {
        printf("Conv_NHWC\n");
        wrapedConv_NHWC_2(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
    } else if (opt ==4) {
        printf("Conv_CHWN\n");
        wrapedConv_CHWN_2(bat4Conv, side, chn, numOfFilter, 1, srcData, filterData_d, &dstData);
    } else if (opt ==5) {
        printf("Conv_CuDNN\n");
        wrapedConv_CuDNN(bat4Conv, side, chn, numOfFilter, padding, srcData, filterData_d, &dstData);
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
