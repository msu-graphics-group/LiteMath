# LiteMath
Lightweight single source math library for graphics without issues

[![Build Status](https://travis-ci.org/richelbilderbeek/travis_qmake_gcc_cpp11_gcov.svg?branch=master)](https://travis-ci.org/richelbilderbeek/coverage2)

## How to get code coverage in HTML

* sudo apt-get install gcovr
* mkdir build && cd build 
* cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON ..
* make -j 8
* make coverage
* see results in 'coverage' directory

