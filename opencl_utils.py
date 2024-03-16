import pyopencl as cl

def load_kernel_source(filename):
    with open(filename, 'r') as file:
        return file.read()

def initialize_opencl():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    return context, queue

def compile_program(context, kernel_source):
    return cl.Program(context, kernel_source).build()

def initialize_buffers(context, grid):
    grid_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)
    new_grid_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, grid.nbytes)
    return grid_buf, new_grid_buf
