nvcc ./CuDNN/convModule_CuDNN.cu -lcudnn -o ./CuDNN/a.out
nvcc ./NCHW/convModule_NCHW_padding.cu -o ./NCHW/a.out
nvcc ./NHWC/convModule_NHWC_padding.cu -o ./NHWC/a.out
nvcc ./CHWN/convModule_CHWN_padding.cu -o ./CHWN/a.out
