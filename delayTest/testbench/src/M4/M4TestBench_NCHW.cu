#include "M4Test.h"

int main(){
    testCase* teCase[9];
    char fileName[] = "../M3/data/gemmOut.bin";

    teCase[0] = new testCase(fileName, 224, 1, 1, 64);      // size, batch, chn
    teCase[1] = new testCase(fileName, 224, 1, 64, 64);

    teCase[2] = new testCase(fileName, 112, 1, 64, 128);
    teCase[3] = new testCase(fileName, 112, 1, 128, 128);

    teCase[4] = new testCase(fileName, 56, 1, 128, 256);
    teCase[5] = new testCase(fileName, 56, 1, 256, 256);

    teCase[6] = new testCase(fileName, 28, 1, 256, 512);
    teCase[7] = new testCase(fileName, 28, 1, 512, 512);

    teCase[8] = new testCase(fileName, 14, 1, 512, 512);

    new0 fun0;
    
    int i=0;
    fun0.valid = fun0.testValid( *teCase[i]);
    fun0.minDelay = fun0.testPerformance( *teCase[i]);

    printf("Fun0: %d, %f\n",fun0.valid, fun0.minDelay);
}