#ifndef __PARA_H__
#define __PARA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

float* get_parameter(const char* filename, int size);

int save_parameter(const char* filename, int size, float *parameter);

#ifdef __cplusplus
}
#endif

#endif
