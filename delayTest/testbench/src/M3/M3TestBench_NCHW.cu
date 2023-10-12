// #pragma once
// #include "util.h"
// #include "testCase.h"
#include "M3Test.h"

int main(){
    testCase* teCase[9];
    char inputTranName[] = "../M1/data/M1_orig.bin";
    char kernelTranName[] ="../M2/data/M2_new0.bin";

    teCase[0] = new testCase(inputTranName, kernelTranName, 224, 1, 1, 64);      // size, batch, chn
    // teCase[1] = new testCase(inputTranName, kernelTranName, 224, 1, 64, 64);

    // teCase[2] = new testCase(inputTranName, kernelTranName, 112, 1, 64, 128);
    // teCase[3] = new testCase(inputTranName, kernelTranName, 112, 1, 128, 128);

    // teCase[4] = new testCase(inputTranName, kernelTranName, 56, 1, 128, 256);
    // teCase[5] = new testCase(inputTranName, kernelTranName, 56, 1, 256, 256);

    // teCase[6] = new testCase(inputTranName, kernelTranName, 28, 1, 256, 512);
    // teCase[7] = new testCase(inputTranName, kernelTranName, 28, 1, 512, 512);

    // teCase[8] = new testCase(inputTranName, kernelTranName, 14, 1, 512, 512);

    new0 fun0;
    
    int i=0;
    fun0.valid = fun0.testValid( *teCase[i]);
    fun0.minDelay = fun0.testPerformance( *teCase[i]);

    printf("Fun0: %d, %f\n",fun0.valid, fun0.minDelay);
}