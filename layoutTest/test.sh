echo "bat4Conv:1  inside:224  chn:128  numOfFilter:128"

echo ""
echo "--------CUDNN Testing--------"
# About the last Argument
# 7 reprensent for non-Fused Wino, 6 represent for fused Wino
./CuDNN/a.out 1 224 128 128 7

echo ""
echo "--------NCHW Testing--------"
./NCHW/a.out 1 224 128 128

echo ""
echo "--------NHWC Testing--------"
./NHWC/a.out 1 224 128 128

echo ""
echo "--------CHWN Testing--------"
./CHWN/a.out 1 224 128 128
