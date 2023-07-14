
#include "GEMM.cuh"
#include "wrapedConv_NCHW.cuh"
// #include "winotrans.cuh"

// function defination

int main(int argc, char** argv){
    const char inputname[] = "../data/input.bin";
    const char filtername[] = "../data/filter.bin";

    int bat4Conv = atoi(argv[1]);
    int inside = atoi(argv[2]);
    int chn = atoi(argv[3]);
    int numOfFilter = atoi(argv[4]);
    int padding = 1;

    //So strange here, why all of this work when I add this "+1", need to figure out !!!
    int nInput = bat4Conv * inside * inside * (chn) +1;
    int nFilter = 9 * numOfFilter * chn;

    float *input_cpu = get_parameter(inputname, nInput);
    float *filter_cpu = get_parameter(filtername, nFilter);
    float *input_gpu, *filter_gpu;  //, *output_gpu;
    cudaMalloc((void **) &input_gpu, nInput<<2);
    cudaMalloc((void **) &filter_gpu, nFilter<<2);
    cudaMemcpy(input_gpu, input_cpu, nInput<<2, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, filter_cpu, nFilter<<2, cudaMemcpyHostToDevice);

    // Output part
    int oside = inside - 2 + 2 * padding;
    int nConvOutput = bat4Conv * oside * oside * numOfFilter;
    float *convOutput_gpu;
    cudaMalloc((void **) &convOutput_gpu, nConvOutput<<2);

    printf("inside: %d, oside:%d\n",inside,oside);
    
    wrapedConv_NCHW(bat4Conv, inside, chn, numOfFilter, padding , input_gpu, filter_gpu,convOutput_gpu);

    float *convOutput_cpu = (float *)malloc(nConvOutput * sizeof(float));
    cudaMemcpy(convOutput_cpu, convOutput_gpu, nConvOutput<<2, cudaMemcpyDeviceToHost);
    
    printf("convOutput_cpu[0]:%lf\n",convOutput_cpu[0]);
    if ( convOutput_cpu[0] == 0.0f) {
        //printf("Test3:\n 1:%f\t 2:%f\n",output_cpu[0], output_cpu[1]);
        printf("moduleConv error\n");
    }

    //used for debug;
    const char finalName[] = "../data/ConvModule_NCHW.bin";
    int cnt4 = save_parameter(finalName, nConvOutput, convOutput_cpu);

    return 0;
}