#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <FreeImage.h>
#include "gemv.h"
#include "gemm.h"
#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

// int save_parameter(const char* filename, int size, float *parameter) {
//     FILE* ptr = fopen(filename,"wb+");

//     if(!ptr){
//         printf("Bad file path: %p, %s\n", ptr, strerror(errno));
//         exit(0);
//     }
//     int cnt = fwrite(parameter,sizeof(float),size,ptr);
//     fclose(ptr);
//     return cnt;
// }

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};

// Conversion from FP64
template <typename T> inline T Convert(double x)
{
    return T(x);
}

// Conversion from FP32
template <typename T> inline T Convert(float x) 
{
    return T(x);
}

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }

    std::cout << "Loading binary file " << fname << std::endl;

    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }

    // conversion
    for (int i = 0; i < size; i++)
    {
        data_h[i] = Convert<value_type>(data_tmp[i]);
    }

    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b, cudaMemcpyHostToDevice) );
}

template <class value_type>
void readImage(const char* filename, value_type* imgData_h, int size) {
    if (!imgData_h) {
      printf("Bad Malloc\n");
      exit(0);
    }
    FILE* ptr = fopen(filename, "rb");

    if (!ptr) {
      printf("Bad file path: %p, %s\n", ptr, strerror(errno));
      exit(0);
    }
    fread(imgData_h,  sizeof(value_type), size, ptr);
    
    fclose(ptr);
}

template <class value_type>
void printDeviceVector(int batch, int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size * batch];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, batch*size*sizeof(value_type), cudaMemcpyDeviceToHost);
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int n=0;n<batch; n++){
        for (int i = 0; i < size; i++)
        {
            std::cout << Convert<real_type>(vec[n*size + i]) << " ";
        }
        std::cout << std::endl;
    }
    delete [] vec;
}


template <class value_type>
struct Layer_t
{
    int inputs;
    int outputs;

    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;

    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0)
    {}

    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
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
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
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

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    // cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;

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

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }

    void destroyHandles()
    {
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
        tensorFormat = CUDNN_TENSOR_NCHW;
        // tensorFormat = CUDNN_TENSOR_NHWC;
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

    // !!!!!!! TODO: addBias and the fullyconnect lack of batch, the size of the dstTensor of FC-layer lack of batch !!!!!!!!
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
        resize(n * dim_y, dstData);

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
        int padA[convDims] = {0,0};
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
        } else {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }

        resize(n*c*h*w, dstData);
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
     
        resize(n*c*h*w, dstData);
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
        resize(n*c*h*w, dstData);

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

        resize(n*c*h*w, dstData);

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
    
        resize(n*c*h*w, dstData);

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

    void classify_example_modified(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2,
                          const int batch, const int chn, const int side)
    {

        // TBD
        // changing the file reading functions;

        int n,c,h,w;
        h = w = side; n = batch; c = chn;

        value_type *srcData = NULL, *dstData = NULL;
        value_type* imgData_h = (value_type*)malloc(side*side*batch *chn * sizeof(value_type));

        readImage(fname, imgData_h, n*c*h*w);

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, side * side * sizeof(value_type) * batch * chn) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    side * side *sizeof(value_type) * batch * chn,
                                    cudaMemcpyHostToDevice) );

        // TBD
		std::cout <<1 <<	std::endl;
		convoluteForward(conv1, n, c, h, w, srcData, &dstData);
		std::cout <<2 <<	std::endl;
		poolForward(n, c, h, w, dstData, &srcData);
		std::cout <<3 <<	std::endl;

        convoluteForward(conv2, n, c, h, w, srcData, &dstData);
		std::cout <<4 <<	std::endl;
        poolForward(n, c, h, w, dstData, &srcData);
		std::cout <<5 <<	std::endl;

        // const char outputName[] = "./data/threeone.bin.mid";
        // int nOutput = 2*800;
        // float *output_cpu = (float*) malloc(nOutput*4);
        // cudaMemcpy(output_cpu, srcData, nOutput<<2, cudaMemcpyDeviceToHost);
        // save_parameter(outputName, nOutput, output_cpu);

        fullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
		std::cout <<6 <<	std::endl;
        activationForward(n, c, h, w, dstData, &srcData);
		std::cout <<7 <<	std::endl;
        lrnForward(n, c, h, w, srcData, &dstData);
		std::cout <<8 <<	std::endl;

        fullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);
		std::cout <<9 <<	std::endl;
        softmaxForward(n, c, h, w, srcData, &dstData);
		std::cout <<10 <<	std::endl;

        //cuDNN and cuBLAS library calls are asynchronous w.r.t. the host.
        // Need a device sync here before copying back the results.
        checkCudaErrors (cudaDeviceSynchronize());
        
        const int max_digits = 10;

        // Take care of half precision
        value_type result[n][max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, n * max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        
        for (int batch =0; batch <n; batch++) 
        {
            int id = 0;
            for (int i = 1; i < max_digits; i++)
            {
                if (Convert<scaling_type>(result[batch][id]) < Convert<scaling_type>(result[batch][i])) {
                    id = i;
                }
            }

            std::cout << "Batch "<< batch <<" Resulting weights from Softmax:" << id << std::endl;
        }

        printDeviceVector(n, c*h*w, dstData);


        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
    }
};

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
    printf( "mnistCUDNN {<options>}\n");
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
        
        n = 2; c =1; side =28;
        assert(side == IMAGE_H); 
        
        
        // The network arichtecture TBD.
        network_t<float> mnist;
        Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
        Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
        Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
        Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);

        
        // cudnnTensorDescriptor_t srcTensorDesc;
        // checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );

        // The convolution algorithm TBD.
        // Set your conv algo.
        mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);

        if(n ==1 ) {
            const char image_name[] = "data/one.bin";
            mnist.classify_example_modified(image_name, conv1, conv2, ip1, ip2, n, c, side);
        } else {
            const char image_name[] = "data/onethree.bin";
            mnist.classify_example_modified(image_name, conv1, conv2, ip1, ip2, n, c, side);
        }

        // checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );

        // mnist.classify_example_modified(image_name, conv1, conv2, ip1, ip2, n, c, side);
        // std::cout << "\nResult of classification: " << i1 << std::endl;

        cudaDeviceReset();
        exit(0);
    }

    displayUsage();
    cudaDeviceReset();

    exit(-1);
}
