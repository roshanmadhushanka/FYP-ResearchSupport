import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# String data array
data = ['A', 'g', 'g']

# Create numpy array according to the string array
lines = numpy.array(data, dtype=str)

# Allocate cuda memory for input data
lines_gpu = cuda.mem_alloc(lines.size * lines.dtype.itemsize)
# Copy data from host to device
cuda.memcpy_htod(lines_gpu, lines)

# Dimensions
blocks = len(data)
thread_per_block = lines.dtype.itemsize
nbr_values = lines.size * lines.dtype.itemsize

# Create destination array
dest = numpy.zeros((nbr_values,), dtype=numpy.int32)

# Allocate cuda memory for output data
dest_gpu = cuda.mem_alloc(dest.size * dest.dtype.itemsize)

# Define cuda function
mod = SourceModule("""
__global__ void process(int **dest, char **line){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    dest[index] = (int *)line[index];
}
""")

# Run GPU model
gpu_func = mod.get_function("process")
gpu_func(dest_gpu, lines_gpu, grid=(blocks, 1), block=(thread_per_block, 1, 1))
cuda.memcpy_dtoh(dest, dest_gpu)
print(dest)