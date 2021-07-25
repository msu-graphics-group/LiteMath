#!/bin/bash
qmake coverage2.pro
make -j 8
./coverage2
mv *.gcda tests
mv *.gcno tests
for filename in `find tests | egrep '\.cpp'`; 
do 
  echo $filename;
  gcov $filename > /dev/null; 
done
mv LiteMath.h.gcov LiteMath.htmp
rm *.gcov
rm *.o
rm tests/*.gcda
rm tests/*.gcno
mv LiteMath.htmp LiteMath.h.gcov 
rm coverage2
rm Makefile
rm .qmake.stash