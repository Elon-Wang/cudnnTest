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
    new3 fun1;
    fun1.testValid(*teCase[0]);
    fun1.testPerformance(*teCase[0]);

    // new1 fun1;
    // new2 fun2;
    // new3 fun3;
    // new4 fun4;
    // new5 fun5;
    // new6 fun6;
    // need to test the performance of the different dims     
    
    // for(int i=0; i<9; i++) {
    //     // test available
    //     // an array to get the availability;
    //     fun1.testValid(*teCase[i]);
    //     fun2.testValid(*teCase[i]);
    //     fun3.testValid(*teCase[i]);
    //     fun4.testValid(*teCase[i]);

    //     // performance testing
    //     fun1.testPerformance(*teCase[i]);
    //     fun2.testPerformance(*teCase[i]);
    //     fun3.testPerformance(*teCase[i]);
    //     fun4.testPerformance(*teCase[i]);

    //     // output the testing result
    //     printf("Fun1: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
    //     printf("Fun2: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
    //     printf("Fun3: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
    //     printf("Fun4: %b,%f,%f,%f\n",fun1.valid, fun1.minTime, fun1.maxTime, fun1.avgTime);
    // }

}