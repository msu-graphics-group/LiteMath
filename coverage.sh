sudo apt-get install gcovr
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON DUSE_STB_IMAGE=ON .. 
make -j 8
make coverage
