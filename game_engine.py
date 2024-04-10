import pyopencl as cl
import glfw
from OpenGL.GL import *
import os
import time
import imgui
from imgui.integrations.glfw import GlfwRenderer
from abc import ABC, abstractmethod


class GameEngine(ABC):
    def __init__(self, width, height, title, cl_file, target_framerate):
        self.window_width = width
        self.window_height = height
        self.target_framerate = target_framerate

        self.window = self.initialize_window(self.window_width, self.window_height, title)
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        self.cl_context, self.cl_queue = self.initialize_opencl()
        self.program = cl.Program(self.cl_context, self.load_file(cl_file)).build()
        self.last_frame_time = glfw.get_time()
        self.delta_time = 0.0
        self.frame_rate = 0

    @staticmethod
    def initialize_window(window_width, window_height, window_title):
        """Initializes glfw window"""
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        window = glfw.create_window(window_width, window_height, window_title, None, None)
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        return window

    @staticmethod
    def initialize_opencl():
        """Initialize openCL, chooses device to be the first GPU detected"""
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
        cl_platform = cl.get_platforms()[0]
        devices = cl_platform.get_devices()
        gpu_devices = [device for device in devices if device.type == cl.device_type.GPU]
        if not gpu_devices:
            print("No GPU device found.")
            exit(1)
        else:
            device = gpu_devices[0]
        print("Using device:", device.name, "Type:", cl.device_type.to_string(device.type))
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        return context, queue

    def run(self):
        """Main loop of game"""
        print("Starting Program...")

        # Vars for tracking frame count and framerate
        frame_count = 0
        second_timer = glfw.get_time()

        # Main Loop
        while not glfw.window_should_close(self.window):
            # Framerate tracking
            current_time = glfw.get_time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Input processing
            self.impl.process_inputs()
            glfw.poll_events()

            # Run update function
            self.update()

            # Clear screen to black
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render function
            self.render()

            # GUI stuff
            imgui.new_frame()
            self.render_gui()
            imgui.end_frame()
            imgui.render()
            self.impl.render(imgui.get_draw_data())

            # Swap window buffers
            glfw.swap_buffers(self.window)

            # Frame rate calculation
            frame_count += 1
            if current_time - second_timer >= 1.0:
                self.frame_rate = frame_count
                frame_count = 0
                second_timer += 1.0

            # Frame rate limiting
            time_to_wait = (1.0 / self.target_framerate) - (glfw.get_time() - current_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        # Cleanup
        self.impl.shutdown()
        glfw.terminate()

    # Helper Functions
    @staticmethod
    def load_file(filename):
        with open(filename, 'r') as file:
            return file.read()

    def initialize_buffers(self, data):
        buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        new_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE, size=buf.size)
        return buf, new_buf

    def initialize_buffer(self, data):
        buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        return buf

    @abstractmethod
    def update(self):
        """Updates before rendering every frame"""
        pass

    @abstractmethod
    def render(self):
        """Renders after the update every frame"""
        pass

    @abstractmethod
    def render_gui(self):
        """Called after rendering, for rendering the GUI"""
        pass
