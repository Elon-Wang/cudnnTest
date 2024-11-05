fileName=result/aug25/ourNCHW.out
rm $fileName

for iter in 1 2 4 8 16 32 64; do
	# ./a.out 64 64 ${iter} 128 | grep time |grep -Eo '[0-9]+(\.[0-9]+)?' 
	# ./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)' 
	a=`./vggCUDNN set batch=${iter} | grep -Eo '[0-9]+(\.[0-9]+) ms' |grep -Eo '[0-9]+(\.[0-9]+)'`
	printf "%f\t" ${a} >> $fileName
    echo "" >> $fileName
done

