#include "M2Test.h"

int main(){

    testCase* teCase[9];
    char fileName =  "../../data/kernel.bin";

    teCase[0] = new testCase(fileName, );   //TODO: more parameters

    new0 fun0;

    int i =0;
    fun0.valid = fun0.testValid(*teCase[i]);
    fun0.minDelay = fun0.testPerformance(*teCase[i]);

    printf("Fun0: %d, %f\n",fun0.valid, fun0.minDelay);
}