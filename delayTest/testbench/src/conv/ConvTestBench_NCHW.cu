#include "ConvTest.h"

void weightedSum(float *testingTime){
    float totalTime =0;
    totalTime = testingTime[0] + testingTime[1]+ testingTime[2]+ testingTime[3] + testingTime[4] + testingTime[5]*3 + testingTime[6] + testingTime[7] *3 + testingTime[8] *4;
    printf("totalTime:%f\n",totalTime);
}

int main(){

    testCase* teCase[9];
    char inputFile[] = "../M1/data";
    char kernelFile[] = "../M2/data";


    teCase[0] = new testCase(inputFile, kernelFile, 224, 1, 1, 64, 1);      // size, batch, chn
    teCase[1] = new testCase(inputFile, kernelFile, 224, 1, 64, 64, 2);

    teCase[2] = new testCase(inputFile, kernelFile, 112, 1, 64, 128, 3);
    teCase[3] = new testCase(inputFile, kernelFile, 112, 1, 128, 128, 4);
    
    teCase[4] = new testCase(inputFile, kernelFile, 56, 1, 128, 256, 5);
    teCase[5] = new testCase(inputFile, kernelFile, 56, 1, 256, 256, 6);
    
    teCase[6] = new testCase(inputFile, kernelFile, 28, 1, 256, 512, 7);
    teCase[7] = new testCase(inputFile, kernelFile, 28, 1, 512, 512, 8);
    
    teCase[8] = new testCase(inputFile, kernelFile, 14, 1, 512, 512, 9);

    float testingTime[9];

    printf("testing fun0:\n");
    for(int i=0; i<9;i++){
        new0 fun0;
        fun0.valid = fun0.testValid(*teCase[i]);
        fun0.testPerformance(*teCase[i]);
        fun0.reportPerformance(*teCase[i]);
        testingTime[i] = fun0.avgDelay;
        // ~fun0();
    }
    weightedSum(testingTime);

    printf("testing fun1:\n");
    for(int i=0; i<9;i++){
        cudnnConv fun1;
        fun1.valid = fun1.testValid(*teCase[i]);
        fun1.testPerformance(*teCase[i]);
        fun1.reportPerformance(*teCase[i]);
        testingTime[i] = fun1.avgDelay;
        // ~fun0();
    }
    weightedSum(testingTime);
}