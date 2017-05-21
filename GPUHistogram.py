import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time

# Random input
input_array = numpy.random.randint(5, size=100)
print(input_array)

# Define cuda function
mod = SourceModule("""
__global__ void histogram(int *hist, int *data){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd( &hist[ data[index] ], 1);
}
""")

