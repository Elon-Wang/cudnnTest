rm result.num

for iter in 1 2 4 8 16 32 64 128; do
	# ./a.out 64 64 ${iter} 128 | grep time |grep -Eo '[0-9]+(\.[0-9]+)?' 
	# ./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)' 
	a=`./a.out ${iter} 22 128 128 | grep -Eo '[0-9]+(\.[0-9]+) us' |grep -Eo '[0-9]+(\.[0-9]+)'`
	printf "%f\t" ${a} >> result.num
done
echo "" >> result.num
