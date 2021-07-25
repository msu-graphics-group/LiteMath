# LiteMath
Lightweight single source math library for graphics without issues

[![codecov.io](https://codecov.io/github/richelbilderbeek/travis_qmake_gcc_cpp11_gcov/coverage.svg?branch=master)](https://codecov.io/github/richelbilderbeek/travis_qmake_gcc_cpp11_gcov?branch=master)

## How to get code coverage in HTML

* sudo apt-get install gcovr
* mkdir build && cd build 
* cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON ..
* make -j 8
* make coverage
* see results in 'coverage' directory

