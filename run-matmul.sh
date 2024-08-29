if [ $# -ne 4 ]; then
  echo "USAGE: ./run-matmul.sh <Matrix Size> <Place> <GPU_Offload_Size> <MKL_Threads[1-4]>"
  exit
fi
sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/pcm/pcm.so:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64 ./matmul $1 $2 $3 $4
