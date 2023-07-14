#pragma once
#include <bits/stdc++.h>


#define Bccum(sum,a,num1,b,num2,c)    \
{    \
    sum[4*a].x = fma(num1[b],num2[c],sum[4*a].x);\
    sum[4*a].y = fma(num1[b],num2[c+1],sum[4*a].y);\
    sum[4*a].z = fma(num1[b],num2[c+2],sum[4*a].z);\
    sum[4*a].w = fma(num1[b],num2[c+3],sum[4*a].w);\
    \
    sum[4*a+1].x = fma(num1[b+1],num2[c],sum[4*a+1].x);\
    sum[4*a+1].y = fma(num1[b+1],num2[c+1],sum[4*a+1].y);\
    sum[4*a+1].z = fma(num1[b+1],num2[c+2],sum[4*a+1].z);\
    sum[4*a+1].w = fma(num1[b+1],num2[c+3],sum[4*a+1].w);\
    \
    sum[4*a+2].x = fma(num1[b+2],num2[c],sum[4*a+2].x);\
    sum[4*a+2].y = fma(num1[b+2],num2[c+1],sum[4*a+2].y);\
    sum[4*a+2].z = fma(num1[b+2],num2[c+2],sum[4*a+2].z);\
    sum[4*a+2].w = fma(num1[b+2],num2[c+3],sum[4*a+2].w);\
    \
    sum[4*a+3].x = fma(num1[b+3],num2[c],sum[4*a+3].x);\
    sum[4*a+3].y = fma(num1[b+3],num2[c+1],sum[4*a+3].y);\
    sum[4*a+3].z = fma(num1[b+3],num2[c+2],sum[4*a+3].z);\
    sum[4*a+3].w = fma(num1[b+3],num2[c+3],sum[4*a+3].w);\
}

#define Tranload(Cache,a,b,reg)    \
{    \
    Cache[b*4][a] = reg.x;\
    Cache[b*4+1][a] = reg.y;\
    Cache[b*4+2][a] = reg.z;\
    Cache[b*4+3][a] = reg.w;\
}

#define Accum(sum,a,num1,b,num2,c)    \
{    \
    sum[4*a].x = fma(num1[b],num2[c].x,sum[4*a].x);\
    sum[4*a].y = fma(num1[b],num2[c].y,sum[4*a].y);\
    sum[4*a].z = fma(num1[b],num2[c].z,sum[4*a].z);\
    sum[4*a].w = fma(num1[b],num2[c].w,sum[4*a].w);\
    \
    sum[4*a+1].x = fma(num1[b+1],num2[c].x,sum[4*a+1].x);\
    sum[4*a+1].y = fma(num1[b+1],num2[c].y,sum[4*a+1].y);\
    sum[4*a+1].z = fma(num1[b+1],num2[c].z,sum[4*a+1].z);\
    sum[4*a+1].w = fma(num1[b+1],num2[c].w,sum[4*a+1].w);\
    \
    sum[4*a+2].x = fma(num1[b+2],num2[c].x,sum[4*a+2].x);\
    sum[4*a+2].y = fma(num1[b+2],num2[c].y,sum[4*a+2].y);\
    sum[4*a+2].z = fma(num1[b+2],num2[c].z,sum[4*a+2].z);\
    sum[4*a+2].w = fma(num1[b+2],num2[c].w,sum[4*a+2].w);\
    \
    sum[4*a+3].x = fma(num1[b+3],num2[c].x,sum[4*a+3].x);\
    sum[4*a+3].y = fma(num1[b+3],num2[c].y,sum[4*a+3].y);\
    sum[4*a+3].z = fma(num1[b+3],num2[c].z,sum[4*a+3].z);\
    sum[4*a+3].w = fma(num1[b+3],num2[c].w,sum[4*a+3].w);\
}

#define Store4x4(reg,b,global,c,k) \
{\
  *((float4 *)(global+c)) = reg[4*b]; \
  *((float4 *)(global+c+k)) = reg[4*b+1]; \
  *((float4 *)(global+c+2*k)) = reg[4*b+2]; \
  *((float4 *)(global+c+3*k)) = reg[4*b+3]; \
}

#define muladd4x4(alpha,reg,beta,reg1,b) \
{\
  reg[4*b].x = alpha*reg[4*b].x + beta*reg1[4*b].x; \
  reg[4*b].y = alpha*reg[4*b].y + beta*reg1[4*b].y; \
  reg[4*b].z = alpha*reg[4*b].z + beta*reg1[4*b].z; \
  reg[4*b].w = alpha*reg[4*b].w + beta*reg1[4*b].w; \
    \
  reg[4*b+1].x = alpha*reg[4*b+1].x + beta*reg1[4*b+1].x; \
  reg[4*b+1].y = alpha*reg[4*b+1].y + beta*reg1[4*b+1].y; \
  reg[4*b+1].z = alpha*reg[4*b+1].z + beta*reg1[4*b+1].z; \
  reg[4*b+1].w = alpha*reg[4*b+1].w + beta*reg1[4*b+1].w; \
  \
  reg[4*b+2].x = alpha*reg[4*b+2].x + beta*reg1[4*b+2].x; \
  reg[4*b+2].y = alpha*reg[4*b+2].y + beta*reg1[4*b+2].y; \
  reg[4*b+2].z = alpha*reg[4*b+2].z + beta*reg1[4*b+2].z; \
  reg[4*b+2].w = alpha*reg[4*b+2].w + beta*reg1[4*b+2].w; \
  \
  reg[4*b+3].x = alpha*reg[4*b+3].x + beta*reg1[4*b+3].x; \
  reg[4*b+3].y = alpha*reg[4*b+3].y + beta*reg1[4*b+3].y; \
  reg[4*b+3].z = alpha*reg[4*b+3].z + beta*reg1[4*b+3].z; \
  reg[4*b+3].w = alpha*reg[4*b+3].w + beta*reg1[4*b+3].w; \
  \
}

#define Load4x4(global,c,k,reg,b) \
{\
  reg[4*b] = *((float4 *)(global+c)); \
  reg[4*b+1] = *((float4 *)(global+c+k)); \
  reg[4*b+2] = *((float4 *)(global+c+2*k)); \
  reg[4*b+3] = *((float4 *)(global+c+3*k)); \
}

float* get_parameter(const char* filename, int size) {
    float* parameter = (float*)malloc(size * 4);
    if (!parameter) {
      printf("Bad Malloc\n");
      exit(0);
    }
    FILE* ptr = fopen(filename, "rb");
  
    if (!ptr) {
      printf("Bad file path: %p, %s\n", ptr, strerror(errno));
      exit(0);
    }
    fread(parameter, size * 4, 1, ptr);
  
    fclose(ptr);
    return parameter;
}

int save_parameter(const char* filename, int size, float *parameter) {
    FILE* ptr = fopen(filename,"wb");

    if(!ptr){
        printf("Bad file path: %p, %s\n", ptr, strerror(errno));
        exit(0);
    }
    int cnt = fwrite(parameter,sizeof(float),size,ptr);
    fclose(ptr);
    return cnt;
}

int gcd(int a, int b){
 return a % b ? gcd(b, a % b) : b;
}

int lcm(int a, int b){
 return a * b / gcd(a, b);
}

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("vggData/") + std::string(fname));
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

    // std::cout << "Loading binary file " << fname << std::endl;

    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    float sum =0;
    // conversion
    for (int i = 0; i < size; i++)
    {
        data_h[i] = Convert<value_type>(data_tmp[i]);
        sum += abs(data_h[i]);
    }

    // printf("sum of file%s is %f\n",fname,sum);
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
        for (int i = 0; i < 15; i++)
        {
            std::cout << Convert<real_type>(vec[n*size + i]) << " ";
        }
        std::cout << std::endl;
    }
    delete [] vec;
}