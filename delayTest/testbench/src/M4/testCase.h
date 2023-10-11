class testCase{
    public:

    int bat4Gemm =36;
    int inside;
    int numOfFilter;
    int bat4Conv;
    int M,N;
    int oside;
    int nGemmOutput;

    char gemmOutputName[60];

    float *gemmOutput_cpu;
    float *gemmOutput_gpu;

    testCase(char *fileName, int bat4Conv, int size, int chn){
        strcpy(gemmOutputName, fileName);
        int padding =1;
        oside = size +2*padding - 2;
        
        nGemmOutput = 36* M*N;
        gemmOutput_cpu = get_parameter(gemmOutputName, nGemmOutput);

        cudaMalloc((void**) &gemmOutput_gpu, nGemmOutput<<2);

        cudaMemcpy(gemmOutput_gpu, gemmOutput_cpu, nGemmOutput<<2, cudaMemcpyHostToDevice);
    }
}