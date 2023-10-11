// #pragma once
// #include "util.h"
// #include "testCase.h"
#include "M3Test.h"

int main(){
    testCase* teCase[9];
    char fileName[] = "../../data/input.bin";

        
    teCase[0] = new testCase(fileName1, Name2, 3600, 1, 1800);      // size, batch, chn

    new0 fun0;
    
    int i=0
    fun0.valid = fun0.testValid( *teCase[i]);
    fun0.minDelay = fun0.testPerformance( *teCase[i]);

    printf("Fun0: %d, %f\n",fun0.valid, fun0.minDelay);
}