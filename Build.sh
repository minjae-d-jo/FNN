#set -e

srcDir=Src
libDir=Lib
binDir=Bin


function build {
	echo -e ":: $1"
	g++ -std=c++11 -Wall \
		-O3 -flto \
		-I /usr/include/eigen3/ -I $srcDir -I $libDir \
		-o $binDir/$1 \
		$srcDir/$1.cpp
}

function testBuild {
	echo -e ":: $1"
	g++ -std=c++11 -Wall \
		-g -fsanitize=address -fno-omit-frame-pointer \
		-lboost_unit_test_framework -lboost_iostreams \
		-I $srcDir -I $libDir \
		-o $binDir/$1 \
		$srcDir/$1.cpp
}

function debugBuild {
	echo -e ":: $1"
	g++ -DKCORE_DEBUG \
		-std=c++11 -Wall \
		-g -fsanitize=address -fno-omit-frame-pointer \
		-lboost_unit_test_framework -lboost_iostreams \
		-I $srcDir -I $libDir \
		-o $binDir/$1 \
		$srcDir/$1.cpp
}

build FNN
# build FNN_BSGD
# build FNN_SGD
# build FNN_original

# build sum_test
# build unary_lambda
# build eigen
# build MNIST
