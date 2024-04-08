echo "bat4Conv:64  inside:22  chn:128  numOfFilter:128"
# echo "--------CUDNN Testing--------"
# ./CuDNN/a.out 64 22 128 128

echo "--------NCHW Testing--------"
./NCHW/a.out 64 22 128 128

echo "--------NHWC Testing--------"
./NHWC/a.out 64 22 128 128

echo "--------CHWN Testing--------"
./CHWN/a.out 64 22 128 128