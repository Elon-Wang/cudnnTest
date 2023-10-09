#include "testCase.h"

class kernelTranMethod{
public:
    int min_numOfFilter;
    int max_numOfFilter;
    int min_chn;
    int max_chn;
    bool testValid(testCase tc);
    float testPerformance(testCase tc);
    virtual void execut(testCase tc)=0 ;
}

bool kernelTranMethod::testValid(testCase tc){
    if(tc.numOfFilter >= min_numOfFilter && tc.numOfFilter <= max_numOfFilter && tc.chn ){

    } else{
        printf("**************test case invalid*************\n");
        return false;
    }
}

float kernelTranMethod::testPerformance(testCase tc) {

}

// original design is assigned as new0;
class new0 : public kernelTranMethod{
public:
    new0(){
        min_numOfFilter = 1;
        max_numOfFilter = 65525;
        min_chn = 1;
        max_chn = 65525;
    }
    virtual void execut(testCase tc) {
        wino_kernel_trans_nchw_suitFor128<<<>>>();
    }
}
