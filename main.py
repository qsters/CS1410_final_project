import pyopencl as cl
import numpy as np
import glfw
from OpenGL.GL import *
import os
import time

def limit_framerate(start_time, target_fps):
    """
    Delays execution to maintain a target framerate.

    Args:
    - start_time: The timestamp when the current frame started (float).
    - target_fps: The desired framerate to maintain (int).
    """
    frame_duration = 1.0 / target_fps
    elapsed = time.time() - start_time
    remaining_time = frame_duration - elapsed
    if remaining_time > 0:
        time.sleep(remaining_time)


def initialize_window(width, height):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "Pixel Grid Visualization", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def initialize_pixel_grid(width, height):
    # Create a binary grid of 0s and 1s
    binary_grid = np.random.randint(2, size=(height, width), dtype=np.uint8)

    # Expand the binary values to 0 or 255 for RGB channels
    pixel_grid = np.empty((height, width, 4), dtype=np.uint8)
    pixel_grid[..., 0:3] = np.expand_dims(binary_grid, axis=-1) * 255  # Set R, G, B channels
    pixel_grid[..., 3] = 255  # Set alpha channel to 255

    return pixel_grid


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
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    # Get the first available platform
    platform = cl.get_platforms()[0]
    # Get the first GPU device available on this platform. If you want to make sure it selects a GPU,
    # you can iterate over devices and check their type
    devices = platform.get_devices()
    gpu_devices = [device for device in devices if device.type == cl.device_type.GPU]

    # If there are no GPU devices, fall back to the first available device
    if not gpu_devices:
        print("No GPU device found. Using the first available device.")
        device = devices[0]
    else:
        device = gpu_devices[0]  # Use the first GPU device

    print("Using device:", device.name, "Type:", cl.device_type.to_string(device.type))

    # Create a context and command queue for the selected device
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    return context, queue

def compile_program(context, kernel_source):
    return cl.Program(context, kernel_source).build()

def initialize_buffers(context, pixel_grid):
    buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pixel_grid)
    new_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=buf.size)
    return buf, new_buf

def main():
    kernel_source = load_kernel_source('Shaders/game_of_life.cl')  # Ensure this shader is adjusted for pixel data
    context, queue = initialize_opencl()
    program = compile_program(context, kernel_source)

    width, height = 512, 512
    window = initialize_window(width, height)
    if not window:
        return

    image_data = initialize_pixel_grid(width, height)
    image_buf, new_image_buf = initialize_buffers(context, image_data)

    texture = initialize_texture(image_data)

    frame_count = 0
    start_time = time.time()

    while not glfw.window_should_close(window):
        frame_start_time = time.time()

        # Execute the shader
        program.game_of_life(queue, (width, height), None, image_buf, new_image_buf, np.uint32(width), np.uint32(height))
        # Copy the updated pixel grid back to the host
        cl.enqueue_copy(queue, image_data, new_image_buf).wait()

        # Update the texture directly with the updated pixel grid

        glClear(GL_COLOR_BUFFER_BIT)

        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

        render_texture()

        # limit_framerate(frame_start_time, 20)

        glfw.swap_buffers(window)
        glfw.poll_events()

        image_buf, new_image_buf = new_image_buf, image_buf

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Every second, display the framerate and reset counters
        if elapsed_time >= 1.0:
            framerate = frame_count / elapsed_time
            print(f"Framerate: {framerate:.2f} FPS")

            # Reset the counter and timer
            frame_count = 0
            start_time = time.time()


    glfw.terminate()

if __name__ == "__main__":
    main()
