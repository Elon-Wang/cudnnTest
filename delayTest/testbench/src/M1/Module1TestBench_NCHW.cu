// #pragma once
// #include "util.h"
// #include "testCase.h"
#include "M1Test.h"

int main(){
    testCase* teCase[9];
    char fileName[] = "../../data/input.bin";
    
    teCase[0] = new testCase(fileName, 224, 1, 1);      // size, batch, chn
    teCase[1] = new testCase(fileName, 224, 1, 64);

    teCase[2] = new testCase(fileName, 112, 1, 64);
    teCase[3] = new testCase(fileName, 112, 1, 128);

    teCase[4] = new testCase(fileName, 56, 1, 128);
    teCase[5] = new testCase(fileName, 56, 1, 256);

    teCase[6] = new testCase(fileName, 28, 1, 256);
    teCase[7] = new testCase(fileName, 28, 1, 512);

    teCase[8] = new testCase(fileName, 14, 1, 512);

    // orig orig_fun;
    // orig_fun.testValid(*teCase[0]);
    // orig_fun.testPerformance(*teCase[0]);
    // new6 fun1;
    // fun1.testValid(*teCase[0]);
    // fun1.testPerformance(*teCase[0]);

    orig fun0;
    new1 fun1;
    new2 fun2;
    new3 fun3;
    new4 fun4;
    new5 fun5;
    new6 fun6;
    // need to test the performance of the different dims     
    
    // for(int i=1; false; i++) {
        int i=8;
        printf("i:%d\n",i);
    //     // test available
    //     // an array to get the availability;
        fun0.valid = fun0.testValid(*teCase[i]);
        fun1.valid = fun1.testValid(*teCase[i]);
        fun2.valid = fun2.testValid(*teCase[i]);
        fun3.valid = fun3.testValid(*teCase[i]);
        fun4.valid = fun4.testValid(*teCase[i]);
        fun5.valid = fun5.testValid(*teCase[i]);
        fun6.valid = fun6.testValid(*teCase[i]);


        // performance testing
        fun0.minDelay = fun0.testPerformance(*teCase[i]);
        fun1.minDelay = fun1.testPerformance(*teCase[i]);
        fun2.minDelay = fun2.testPerformance(*teCase[i]);
        fun3.minDelay = fun3.testPerformance(*teCase[i]);
        fun4.minDelay = fun4.testPerformance(*teCase[i]);
        fun5.minDelay = fun5.testPerformance(*teCase[i]);
        fun6.minDelay = fun6.testPerformance(*teCase[i]);

        // output the testing result
        printf("Fun0: %d, %f\n",fun0.valid, fun0.minDelay);
        printf("Fun1: %d, %f\n",fun1.valid, fun1.minDelay);
        printf("Fun2: %d, %f\n",fun2.valid, fun2.minDelay);
        printf("Fun3: %d, %f\n",fun3.valid, fun3.minDelay);
        printf("Fun4: %d, %f\n",fun4.valid, fun4.minDelay);
        printf("Fun5: %d, %f\n",fun5.valid, fun5.minDelay);
        printf("Fun6: %d, %f\n",fun6.valid, fun6.minDelay);
    // }

}