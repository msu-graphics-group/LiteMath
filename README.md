# LiteMath
Lightweight single source math library for graphics without issues

[![codecov.io](https://codecov.io/gh/msu-graphics-group/LiteMath/coverage.svg)](https://codecov.io/gh/msu-graphics-group/LiteMath)

## How to get code coverage in HTML

* sudo apt-get install gcovr
* mkdir build && cd build 
* cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON ..
* make -j 8
* make coverage
* see results in 'coverage' directory

