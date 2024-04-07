import pyopencl as cl
import glfw
from OpenGL.GL import *
import os
import time
import imgui
from imgui.integrations.glfw import GlfwRenderer

class GameEngine:
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

    def initialize_window(self, window_width, window_height, window_title):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
            exit(1)

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
            exit(1)

        return window



    def initialize_opencl(self):
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
        platform = cl.get_platforms()[0]
        devices = platform.get_devices()
        gpu_devices = [device for device in devices if device.type == cl.device_type.GPU]
        if not gpu_devices:
            print("No GPU device found. Using the first available device.")
            device = devices[0]
        else:
            device = gpu_devices[0]
        print("Using device:", device.name, "Type:", cl.device_type.to_string(device.type))
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        return context, queue

    def run(self):
        print("Starting Program...")
        frame_count = 0
        second_timer = glfw.get_time()
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.impl.process_inputs()

            glfw.poll_events()

            self.update()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.render()

            imgui.new_frame()
            self.render_gui()
            imgui.end()
            imgui.pop_style_var()
            imgui.render()

            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            self.post_render()

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

        self.impl.shutdown()
        glfw.terminate()

    # Helper Functions
    def load_file(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def initialize_buffers(self, data):
        buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        new_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE, size=buf.size)
        return buf, new_buf

    def initialize_buffer(self, data):
        buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        return buf

    # Methods meant to be overridden
    def update(self):
        """Updates before rendering every frame"""
        pass

    def render(self):
        """Renders after the update every frame"""
        pass

    def post_render(self):
        """Called once after rendering"""
        pass

    def render_gui(self):
        """Called after rendering, for rendering the GUI"""
        pass
