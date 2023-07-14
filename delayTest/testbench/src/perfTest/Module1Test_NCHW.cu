#include "util.h"

__global__ void warmup(){}
__global__ void wino_input_trans_nchw_suitFor128(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);
__global__ void wino_input_trans_nchw_suitFor128_new3(int side, int side_beta, int MSize, int KSize, int padding, float * pInputs, float* pOutputs);


int main(){
    // need to test the performance of the different dims     
    
    new1 fun1();
    new2 fun2();
    new3 fun3();
    new4 fun4();


    // test available
    // an array to get the availability;
    fun1.testValid();
    fun2.testValid();
    fun3.testValid();
    fun4.testValid();

    // get performance
    

    // output the testing result

}

class inputTransMethod{
    int minSide;
    int maxSide;
    int maxChannel;
    int maxBatch;

    // counted in micro second
    float minDelay;
    float maxDelay;
    float avgDelay;
    bool valid= false;
    bool testValid(int batch, int channel, int inside);

    void testPerf(int batch, int channel, int inside);

    inputTransMethod(){
        
    }
}

class new1 : public inputTransMethod{
    
}

class new2 : public inputTransMethod{
    
}

class new3 : public inputTransMethod{
    
}

class new4 : public inputTransMethod{
    
}