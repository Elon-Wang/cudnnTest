file_Name=result/sep20/layre9.out
rm $file_Name

echo "NCHW" >> $file_Name
for iter in 1 2 4 8 16 32 64; do
	# for opt in 1 2 3; do 
		# ./a.out 64 64 ${iter} 128 | grep time |grep -Eo '[0-9]+(\.[0-9]+)?' 
		# ./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)' 
		a=`./a.out batch=${iter} opt=2 | grep -Eo '[0-9]+(\.[0-9]+) ms' |grep -Eo '[0-9]+(\.[0-9]+)'`
		# a=`./a.out batch=${iter} opt=3 |grep -Eo '[0-9]+\.[0-9]+ ms\s+[ 0-9\.\,\(\)]+'|grep -Eo '[0-9]+\.[0-9]+'`
		printf "%f\t" ${a} >> $file_Name
		# echo "" >> $file_Name
	# done 
	echo "" >> $file_Name
done

echo "NHWC" >> $file_Name
for iter in 1 2 4 8 16 32 64; do
	# for opt in 1 2 3; do 
		# ./a.out 64 64 ${iter} 128 | grep time |grep -Eo '[0-9]+(\.[0-9]+)?' 
		# ./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)' 
		a=`./a.out batch=${iter} opt=3 | grep -Eo '[0-9]+(\.[0-9]+) ms' |grep -Eo '[0-9]+(\.[0-9]+)'`
		# a=`./a.out batch=${iter} opt=3 |grep -Eo '[0-9]+\.[0-9]+ ms\s+[ 0-9\.\,\(\)]+'|grep -Eo '[0-9]+\.[0-9]+'`
		printf "%f\t" ${a} >> $file_Name
		# echo "" >> $file_Name
	# done 
	echo "" >> $file_Name
done

echo "CHWN" >> $file_Name
for iter in 1 2 4 8 16 32 64; do
	# for opt in 1 2 3; do 
		# ./a.out 64 64 ${iter} 128 | grep time |grep -Eo '[0-9]+(\.[0-9]+)?' 
		# ./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)' 
		a=`./a.out batch=${iter} opt=4 | grep -Eo '[0-9]+(\.[0-9]+) ms' |grep -Eo '[0-9]+(\.[0-9]+)'`
		# a=`./a.out batch=${iter} opt=3 |grep -Eo '[0-9]+\.[0-9]+ ms\s+[ 0-9\.\,\(\)]+'|grep -Eo '[0-9]+\.[0-9]+'`
		printf "%f\t" ${a} >> $file_Name
		# echo "" >> $file_Name
	# done 
	echo "" >> $file_Name
done