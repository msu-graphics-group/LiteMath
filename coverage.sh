sudo apt-get install gcovr
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON ..
make -j 8
make coverage
