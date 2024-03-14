import pyopencl as cl
import numpy as np
import glfw
from OpenGL.GL import *
import time


def initialize_window(width, height):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "Binary Grid Visualization", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def convert_to_pixel_grid(binary_grid):
    pixel_grid = binary_grid * 255
    pixel_grid_rgba = np.stack([pixel_grid]*4, axis=-1).astype(np.uint8)
    pixel_grid_rgba[..., 3] = 255  # Set alpha to 255
    return pixel_grid_rgba

def initialize_texture(pixel_grid):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pixel_grid.shape[1], pixel_grid.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel_grid)
    return texture

def render_texture():
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(-1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)

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

def main():
    kernel_source = load_kernel_source('Shaders/game_of_life.cl')
    context, queue = initialize_opencl()
    program = compile_program(context, kernel_source)

    # Window and OpenGL initialization
    width, height = 512, 512
    window = initialize_window(width, height)
    if not window:
        return

    # Initialize the game grid as a binary grid
    grid = np.random.randint(2, size=(height, width), dtype=np.int32)
    grid_buf, new_grid_buf = initialize_buffers(context, grid)

    texture = initialize_texture(grid)

    while not glfw.window_should_close(window):
        # Execute the Game of Life kernel
        program.game_of_life(queue, (width, height), None, grid_buf, new_grid_buf, np.uint32(width), np.uint32(height))

        # Copy the updated grid back to the host
        cl.enqueue_copy(queue, grid, new_grid_buf).wait()

        # Convert the binary grid to a pixel grid for rendering
        pixel_grid = convert_to_pixel_grid(grid)

        # Update the texture with the pixel grid
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_grid)

        # Render the texture
        glClear(GL_COLOR_BUFFER_BIT)
        render_texture()
        glfw.swap_buffers(window)
        glfw.poll_events()

        # Swap the buffers for the next iteration
        grid_buf, new_grid_buf = new_grid_buf, grid_buf

    glfw.terminate()

if __name__ == "__main__":
    main()