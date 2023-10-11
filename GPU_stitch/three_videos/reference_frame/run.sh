file='cuda_three_stitch_reference.cpp'
g++ $file -o out `pkg-config --cflags --libs opencv`
./out 
rm ./out