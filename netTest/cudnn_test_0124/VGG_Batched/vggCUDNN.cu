#include <bits/stdc++.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include "error_util.h"
#include "util.h"
#include "Layer.h"
#include "Network.h"

#define IMAGE_H 224
#define IMAGE_W 224

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *conv3_bin = "conv3.bin";
const char *conv3_bias_bin = "conv3.bias.bin";
const char *conv4_bin = "conv4.bin";
const char *conv4_bias_bin = "conv4.bias.bin";
const char *conv5_bin = "conv5.bin";
const char *conv5_bias_bin = "conv5.bias.bin";
const char *conv6_bin = "conv6.bin";
const char *conv6_bias_bin = "conv6.bias.bin";
const char *conv7_bin = "conv7.bin";
const char *conv7_bias_bin = "conv7.bias.bin";
const char *conv8_bin = "conv8.bin";
const char *conv8_bias_bin = "conv8.bias.bin";
const char *conv9_bin = "conv9.bin";
const char *conv9_bias_bin = "conv9.bias.bin";
const char *conv10_bin = "conv10.bin";
const char *conv10_bias_bin = "conv10.bias.bin";
const char *conv11_bin = "conv11.bin";
const char *conv11_bias_bin = "conv11.bias.bin";
const char *conv12_bin = "conv12.bin";
const char *conv12_bias_bin = "conv12.bias.bin";
const char *conv13_bin = "conv13.bin";
const char *conv13_bias_bin = "conv13.bias.bin";
const char *fc14_bin = "fc14.bin";
const char *fc14_bias_bin = "fc14.bias.bin";
const char *fc15_bin = "fc15.bin";
const char *fc15_bias_bin = "fc15.bias.bin";
const char *fc16_bin = "fc16.bin";
const char *fc16_bias_bin = "fc16.bias.bin";

static char * baseFile(char *fname) 
{
    char *base;
    for (base = fname; *fname != '\0'; fname++) {
        if (*fname == '/' || *fname == '\\') {
            base = fname + 1;
        }
    }
    return base;
}

static void displayUsage()
{
    printf( "vggCUDNN {<options>}\n");
    printf( "help                   : display this help\n");
    printf( "device=<int>           : set the device to run the sample\n");
    printf( "image=<name>           : classify specific image\n");
    printf( "set=<name>             : a set of images more than one batch\n");
    printf( "n,c,h,w=<int>          : specify the parameter of the test\n");
}

int main(int argc, char *argv[]) 
{   
    std::string image_path;
    // int i1,i2,i3;

    printf("Executing: %s", baseFile(argv[0]));
    for (int i = 1; i < argc; i++) {
        printf(" %s", argv[i]);
    }
    printf("\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayUsage();
        exit(-1); 
    }

    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\n", COMPILER_NAME, COMPILER_VER);
    showDevices();

    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
    }
    std::cout << "Using device " << device << std::endl;

    if (checkCmdLineFlag(argc, (const char **)argv, "set"))
    {
        //char* image_name;
        int n,c,side;
        // getCmdLineArgumentString(argc, (const char **)argv,
        //                          "set", (char **) &image_name);
        // n = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
        // c = getCmdLineArgumentInt(argc, (const char **)argv, "chn");
        // side = getCmdLineArgumentInt(argc, (const char **)argv, "side");
        
        n = 1; c =3; side =224;
        assert(side == IMAGE_H); 
        
        
        // The network arichtecture TBD.
        network_t<float> vgg16;
        // Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
        // Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
        // Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
        // Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);

        Layer_t<float>  conv1(   3,  64,3,conv1_bin,conv1_bias_bin,argv[0]);
        Layer_t<float>  conv2(  64,  64,3,conv2_bin,conv2_bias_bin,argv[0]);

        Layer_t<float>  conv3(  64, 128,3,conv3_bin,conv3_bias_bin,argv[0]);
        Layer_t<float>  conv4( 128, 128,3,conv4_bin,conv4_bias_bin,argv[0]);

        Layer_t<float>  conv5( 128, 256,3,conv5_bin,conv5_bias_bin,argv[0]);
        Layer_t<float>  conv6( 256, 256,3,conv6_bin,conv6_bias_bin,argv[0]);
        Layer_t<float>  conv7( 256, 256,3,conv7_bin,conv7_bias_bin,argv[0]);

        Layer_t<float>  conv8( 256, 512,3,conv8_bin,conv8_bias_bin,argv[0]);
        Layer_t<float>  conv9( 512, 512,3,conv9_bin,conv9_bias_bin,argv[0]);
        Layer_t<float> conv10( 512, 512,3,conv10_bin,conv10_bias_bin,argv[0]);

        Layer_t<float> conv11( 512, 512,3,conv11_bin,conv11_bias_bin,argv[0]);
        Layer_t<float> conv12( 512, 512,3,conv12_bin,conv12_bias_bin,argv[0]);
        Layer_t<float> conv13( 512, 512,3,conv13_bin,conv13_bias_bin,argv[0]);

        Layer_t<float>   fc14(25088,4096,1,fc14_bin, fc14_bias_bin,argv[0]);
        Layer_t<float>   fc15(4096,4096,1, fc15_bin, fc15_bias_bin,argv[0]);
        Layer_t<float>   fc16(4096,1000,1, fc16_bin, fc16_bias_bin,argv[0]);

        // The convolution algorithm TBD.
        // Set your conv algo.
        // vgg16.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
        vgg16.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
        
        // stange things, why change the data format wont't impact the accuracy?
        vgg16.setTensorFormat(CUDNN_TENSOR_NCHW);
        // vgg16.setTensorFormat(CUDNN_TENSOR_NHWC);

        // float time[1000];
        int result[1000];
        std::vector<int> ret;
        int err = 0;
        int Round = 10;
        warmup<<<1,1>>>();
        n =64;

        for(int i=0; i< Round;i++) {
            
            if (i %50 ==0)
                std::cout << "Performing forward propagation "<<  (100*i/(float)Round) <<"% ...\n";
            // Batched Conv testing
            // char file_name[30] = "./binImage/Batch2/bat2_";
            char file_name[40]; 
            sprintf(file_name, "./binImage/Batch%d/bat%d_%d.bin", n, n, i);

            // char str2[5];
            // const char str3[] = ".bin";
            // std::sprintf(str2, "%d", i);
            // std::strcat(file_name, str2);
            // std::strcat(file_name, str3);
            // std::cout<< image_name;
            ret = vgg16.classify_example_modified(file_name, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, fc14, fc15, fc16, n, c, side); 

            for(int j=0; j<n; j++) {
                err += (ret[j] == (i*n+j) ? 1:0);
                result[i*n + j] = ret[j];
            }
            
            // multi-batch testing
            // const char image_name[] = "vggData/img0.bin";
            //vgg16.classify_example_modified(image_name, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, fc14, fc15, fc16, n, c, side); 
        }

        // print out the result;
        printf("Acc: %f %%\n", (100*err/(float)(Round*n)));
        const char* filename = "result/CNNprediction.bin";
        const char* changeLine = "\n"; 
        FILE* ptr = fopen(filename,"w");
        fclose(ptr);
        ptr = fopen(filename,"a");
        for(int i=0;i<Round*n;i++) {
            fwrite((result+i),sizeof(int),1,ptr);
            fwrite(changeLine, sizeof(char),1,ptr);
        }

        // std::cout << "\nResult of classification: " << i1 << std::endl;

        cudaDeviceReset();
        exit(0);
    }

    displayUsage();
    cudaDeviceReset();

    exit(-1);
}
