# LiteMath
Lightweight single source math library for graphics without issues

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![codecov](https://codecov.io/gh/msu-graphics-group/LiteMath/branch/dev/graph/badge.svg?token=KG13KA0LFV)](https://app.codecov.io/gh/msu-graphics-group/LiteMath/tree/dev)

# LiteImage
Lightweight image implementation for most common tasts: mainly for algorithms prototyping.
* Plain, simple and stupid as hammer
* pitch-linear data layout
* Close-to-shaders usage sematics
* understandable by [kernel_slicer](https://github.com/Ray-Tracing-Systems/kernel_slicer), i.e. support automatic translation from C++ to GLSL
* Native load/stote support for '.ppm','.bmp' and plain binary formats (line '.image4f') 
* Optional load/stote support for '.png' and '.jpg' via stb_image (-DUSE_STB_IMAGE=ON)

# Examples
TBDD

## How to get code coverage in HTML

* sudo apt-get install gcovr
* mkdir build && cd build 
* cmake -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=ON .. -DUSE_STB_IMAGE=ON
* make -j 8
* make coverage
* see results in 'coverage' directory
* Image2d.h/Image2d.cpp are excluded from the report for now, it measures LiteMath.h only

