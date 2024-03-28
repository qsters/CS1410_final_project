import pyopencl as cl
import numpy as np
import glfw
from OpenGL.GL import *
import os
import time

class GameEngine:
    def __init__(self, width, height, title, cl_file):
        self.width = width
        self.height = height
        self.window_title = title
        self.window = None
        self.initialize_window()
        self.cl_context, self.cl_queue = self.initialize_opencl()
        self.program = cl.Program(self.cl_context, self.load_kernel_source(cl_file)).build()

    def initialize_window(self):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        self.window = glfw.create_window(self.width, self.height, self.window_title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        print("Window Initialized...")

    def initialize_opencl(self):
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
        platform = cl.get_platforms()[0]
        devices = platform.get_devices()
        gpu_devices = [device for device in devices if device.type == cl.device_type.GPU]

        # If there are no GPU devices, fall back to the first available device
        if not gpu_devices:
            print("No GPU device found. Using the first available device.")
            device = devices[0]
        else:
            device = gpu_devices[0]  # Use the first GPU device

        print("Using device:", device.name, "Type:", cl.device_type.to_string(device.type))

        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        return context, queue


    def run(self):
        print("Starting Program...")
        while not glfw.window_should_close(self.window):
            self.update()
            glClear(GL_COLOR_BUFFER_BIT)
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self.post_render()

        glfw.terminate()

    # Helper Functions
    def load_kernel_source(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def initialize_buffers(self, data):
        buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        new_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE, size=buf.size)
        return buf, new_buf

    # Methods meant to be overriden
    def update(self):
        """Updates before rendering every frame"""
        pass

    def render(self):
        """Renders after the update every frame"""
        pass

    def post_render(self):
        """Called once after rendering"""
        pass

class ConwaysGameOfLife(GameEngine):
    def __init__(self, width, height, title='Conway\'s Game of Life', cl_file="Shaders/game_of_life.cl"):
        super().__init__(width, height, title, cl_file)
        self.image_data = self.initialize_pixel_grid(self.width, self.height)
        self.image_buf, self.new_image_buf = self.initialize_buffers(self.image_data)
        self.texture = self.initialize_texture()

    def initialize_pixel_grid(self, width, height):
        binary_grid = np.random.randint(2, size=(height, width), dtype=np.uint8)
        pixel_grid = np.empty((height, width, 4), dtype=np.uint8)
        pixel_grid[..., 0:3] = np.expand_dims(binary_grid, axis=-1) * 255
        pixel_grid[..., 3] = 255
        return pixel_grid

    def initialize_texture(self):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_data.shape[1], self.image_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)
        return texture

    def render_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def render(self):
        self.render_texture()

    def post_render(self):
        self.image_buf, self.new_image_buf = self.new_image_buf, self.image_buf

    def update(self):
        self.program.game_of_life(self.cl_queue, (self.width, self.height), None, self.image_buf, self.new_image_buf, np.uint32(self.width), np.uint32(self.height))
        cl.enqueue_copy(self.cl_queue, self.image_data, self.new_image_buf).wait()

game = ConwaysGameOfLife(512, 512)
game.run()