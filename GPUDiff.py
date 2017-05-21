import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time


def asciiDiff(word1, word2):
    lst = numpy.zeros((size_arr,), dtype=numpy.int32)
    for i in range(len(word1)):
        lst[i] = ord(word1[i]) - ord(word2[i])
    return lst

# String data array
f1 = open('text1', 'r')
f2 = open('text2', 'r')
word1 = f1.read()
word2 = f2.read()
f1.close()
f2.close()

# Display
print("Process Size : " + str(len(word1)) + " number of characters")

if len(word1) > len(word2):
    word2 = word2.ljust(len(word1), '*')
else:
    word1 = word1.ljust(len(word2), '*')


# Define cuda function
mod = SourceModule("""
__global__ void process(int *dest, char *line1, char *line2){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    dest[index] = (int)line1[index] - (int)line2[index];
}
""")

start = time.clock()
# Create numpy array according to the string array
lines1 = numpy.array(list(word1), dtype=str)
lines2 = numpy.array(list(word2), dtype=str)

# Dimensions
size_arr = lines1.size * lines1.dtype.itemsize

# Allocate cuda memory for input data
lines1_gpu = cuda.mem_alloc(size_arr)
lines2_gpu = cuda.mem_alloc(size_arr)

# Copy data from host to device
cuda.memcpy_htod(lines1_gpu, lines1)
cuda.memcpy_htod(lines2_gpu, lines2)

# Dimensions
thread_per_block = 300
blocks = int(lines1.size / thread_per_block) + 1

dest = numpy.zeros((size_arr,), dtype=numpy.int32)


# Allocate cuda memory for output data
dest_gpu = cuda.mem_alloc(dest.size * dest.dtype.itemsize)

# Run GPU model
gpu_func = mod.get_function("process")
gpu_func(dest_gpu, lines1_gpu, lines2_gpu, grid=(blocks, 1), block=(thread_per_block, 1, 1))
cuda.memcpy_dtoh(dest, dest_gpu)
print(dest)
print(time.clock() - start)

start = time.clock()
print(asciiDiff(word1, word2))
print(time.clock() - start)