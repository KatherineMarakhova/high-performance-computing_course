#!/bin/bash

#compile
gcc -lstdc++ -fopenmp parallel.cpp -o parallel.out
./parallel.out
