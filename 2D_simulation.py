import math

import imgui
import pyopencl as cl
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine

class Simulation2D(GameEngine):
    def __init__(self, width, height, spore_count=300, title="Slime Mold Sim 2D", target_framerate=60):
        super().__init__(width, height, title, "Shaders/2d_simulation.cl", target_framerate)

        self.image_data = self.get_empty_image()
        self.image_buf, _ = self.initialize_buffers(self.image_data)
        self.texture = self.initialize_texture()

        self.fragment_shader = self.load_file("Shaders/fragment_shader.glsl")
        self.vertex_shader = self.load_file("Shaders/vertex_shader.glsl")

        self.spore_count = spore_count
        self.spores = self.initialize_spores()
        self.spores_buffer = self.initialize_buffer(self.spores)

        self.random_seeds = np.random.randint(0, 2**32 - 1, size=self.spore_count, dtype=np.uint32)
        self.random_seeds_buffer = self.initialize_buffer(self.random_seeds)

        self.spore_speed = 100
        self.decay_speed = .2
        self.sensor_distance = 5
        self.turn_speed = 3

        self.decay_accumulator = 0

        self.settings = self.create_settings()
        self.settings_buffer = self.initialize_buffer(self.settings)

        self.shader_program = self.create_shader_program()
        self.setup_rendering()  # Call after creating the shader program


    def create_settings(self):
        settings_dtype = np.dtype([
            ('spore_count', np.uint32),
            ('screen_height', np.uint32),
            ('screen_width', np.uint32),
            ('spore_speed', np.float32),
            ('decay_speed', np.float32),
            ('turn_speed', np.float32),
            ('sensor_distance', np.float32)
        ])

        settings = np.array([(self.spore_count, self.window_height, self.window_width, self.spore_speed, self.decay_speed, self.turn_speed,self.sensor_distance)], dtype=settings_dtype)
        return settings

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode('utf-8')
            raise RuntimeError("Shader compilation error: " + error)
        return shader

    def create_shader_program(self):
        vertex_shader = self.compile_shader(self.vertex_shader, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.fragment_shader, GL_FRAGMENT_SHADER)

        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)

        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(shader_program).decode('utf-8')
            raise RuntimeError("Shader link error: " + error)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def initialize_texture(self):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_data.shape[1], self.image_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)
        return texture

    def setup_rendering(self):
        # Vertex data: positions and texture coordinates for a fullscreen quad
        vertex_data = np.array([
            -1.0, -1.0,  0.0, 0.0,  # Triangle 1, Bottom-left
            1.0, -1.0,  1.0, 0.0,  # Triangle 1, Bottom-right
            -1.0,  1.0,  0.0, 1.0,  # Triangle 1, Top-left
            1.0, -1.0,  1.0, 0.0,  # Triangle 2, Bottom-right
            -1.0,  1.0,  0.0, 1.0,  # Triangle 2, Top-left
            1.0,  1.0,  1.0, 1.0   # Triangle 2, Top-right
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertex_data.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertex_data.itemsize, ctypes.c_void_p(2 * vertex_data.itemsize))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def render_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.window_width, self.window_height, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)

        glBindVertexArray(self.vao)
        # Use your shader program
        glUseProgram(self.shader_program)
        # Now draw the quad
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

    def get_empty_image(self):
        pixel_grid = np.empty((self.window_height, self.window_width, 4), dtype=np.uint8)
        pixel_grid[..., 3] = 255
        return pixel_grid

    def initialize_spores(self):
        # Define the structured datatype for a 2D spore
        spore_dtype = np.dtype([
            ('x', np.float32),  # x position
            ('y', np.float32),  # y position
            ('angle', np.float32),  # y position
        ])

        # Initialize empty array of spores
        spores = np.zeros(self.spore_count, dtype=spore_dtype)

        # Randomize positions within the bounds of height and width
        spores['x'] = np.random.uniform(0, self.window_width, size=self.spore_count)
        spores['y'] = np.random.uniform(0, self.window_height, size=self.spore_count)
        spores['angle'] = np.random.uniform(0, 2 * math.pi, size=self.spore_count)

        # spores['x'] = self.window_width / 2
        # spores['y'][0] = self.window_height / 2
        # spores['angle'][0] = 0

        return spores

    def update(self):
        # Calculate the total decay amount to add to the accumulator
        self.decay_accumulator += self.decay_speed * self.delta_time * 255

        # Extract the whole number part of the decay amount to apply
        decay_amount = int(self.decay_accumulator)

        # Update the accumulator to retain only the fractional part
        self.decay_accumulator -= decay_amount

        self.program.fade_image(self.cl_queue, (self.window_width, self.window_height), None, self.image_buf, self.settings_buffer, np.uint32(decay_amount))
        self.program.draw_spores(self.cl_queue, (self.spore_count,), None, self.image_buf, self.spores_buffer, self.settings_buffer)
        self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.image_buf, self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        cl.enqueue_copy(self.cl_queue, self.image_data, self.image_buf).wait()

    def render(self):
        # Activate the shader program
        glUseProgram(self.shader_program)

        # Bind the texture to texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.window_width, self.window_height, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)


        # Tell the shader the texture is in texture unit 0
        glUniform1i(glGetUniformLocation(self.shader_program, "texture1"), 0)

        # Bind the VAO (and therefore the VBO and attribute configurations)
        glBindVertexArray(self.vao)

        # Draw the quad (2 triangles, 6 vertices)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Unbind the VAO and shader program
        glBindVertexArray(0)
        glUseProgram(0)
    def render_gui(self):

        # Set the window flags
        window_flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE

        # Set the window's background alpha (transparency) to 0.7 (1.0 is opaque, 0.0 is transparent)
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.8)

         # Start a new ImGui window
        if imgui.begin("Simulation Parameters"):

            # Slider for spore_speed
            changed, self.spore_speed = imgui.slider_float("Spore Speed", self.spore_speed, 5.0, 200.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for decay_speed
            changed, self.decay_speed = imgui.slider_float("Decay Speed", self.decay_speed, 0.1, 1.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for sensor_distance
            changed, self.sensor_distance = imgui.slider_float("Sensor Distance", self.sensor_distance, 0.1, 10.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for turn_speed
            changed, self.turn_speed = imgui.slider_float("Turn Speed", self.turn_speed, 0.0, 5.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

    def update_settings_buffer(self):
        # Create a new settings array
        settings = np.array([(self.spore_count, self.window_height, self.window_width,
                              self.spore_speed, self.decay_speed, self.turn_speed,
                              self.sensor_distance)], dtype=self.create_settings().dtype)

        # Update the buffer with the new settings
        cl.enqueue_copy(self.cl_queue, self.settings_buffer, settings)

if __name__ == '__main__':
    game = Simulation2D(500, 500, 1000, target_framerate=100)
    game.run()
