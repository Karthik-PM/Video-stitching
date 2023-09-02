file=$1
g++ $1 -o out `pkg-config --cflags --libs opencv`
./out 
rm ./out