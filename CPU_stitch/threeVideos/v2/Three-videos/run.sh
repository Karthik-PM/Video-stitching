file='threeVideoStitching.cpp'
g++ $file -o out `pkg-config --cflags --libs opencv`
./out 
rm ./out