import math

import imgui
import pyopencl as cl
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine
import glm

class Simulation3D(GameEngine):
    def __init__(self, window_width, window_height, simulation_width=10, simulation_height=10, simulation_length=10, spore_count=300, title="Slime Mold Sim 2D", target_framerate=60):
        super().__init__(window_width, window_height, title, "Shaders/2d_simulation.cl", target_framerate)

        self.simulation_width = simulation_width
        self.simulation_height = simulation_height
        self.simulation_length = simulation_length

        self.volume_data = self.get_empty_volume()
        self.instance_positions = self.get_instance_positions()
        self.instance_vbo = self.get_instance_position_buffer()
        self.image_buf, _ = self.initialize_buffers(self.volume_data)
        self.texture = self.initialize_texture()

        self.fragment_shader = self.load_file("Shaders/3D_fragment_shader.glsl")
        self.vertex_shader = self.load_file("Shaders/3D_vertex_shader.glsl")

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
        self.vao, self.vbo, self.ebo = self.setup_rendering()
        self.projection, self.view, self.model, self.model_loc, self.view_loc, self.proj_loc = self.setup_matrices()
        glEnable(GL_DEPTH_TEST)  # Enable depth test


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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.volume_data.shape[1], self.volume_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.volume_data)
        return texture

    def setup_rendering(self):
        # Vertex data: positions and texture coordinates for a fullscreen quad
        # Define the 8 vertices of the cube
        vertices = np.array([
            -0.5, -0.5, -0.5,  # Vertex 0
            0.5, -0.5, -0.5,  # Vertex 1
            0.5,  0.5, -0.5,  # Vertex 2
            -0.5,  0.5, -0.5,  # Vertex 3
            -0.5, -0.5,  0.5,  # Vertex 4
            0.5, -0.5,  0.5,  # Vertex 5
            0.5,  0.5,  0.5,  # Vertex 6
            -0.5,  0.5,  0.5   # Vertex 7
        ], dtype=np.float32)

        # Define the 12 triangles (2 per face) that make up the cube
        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front face
            1, 5, 6, 6, 2, 1,  # Right face
            7, 6, 5, 5, 4, 7,  # Back face
            4, 0, 3, 3, 7, 4,  # Left face
            4, 5, 1, 1, 0, 4,  # Bottom face
            3, 2, 6, 6, 7, 3   # Top face
        ], dtype=np.uint32)

        # Create and bind the Vertex Array Object
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Create and bind the Vertex Buffer Object
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Create and bind the Element Buffer Object
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Enable the vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao, vbo, ebo

    def get_instance_positions(self):
        positions = []
        # Iterate through the volume data to find voxels set to 255
        for z in range(self.simulation_length):
            for y in range(self.simulation_height):
                for x in range(self.simulation_width):
                    if self.volume_data[y, x, z] == 255:
                        # Convert grid coordinates to world space coordinates as needed
                        positions.append([x, y, z])
        # Convert to a numpy array for easier handling
        return np.array(positions, dtype=np.float32)

    def get_instance_position_buffer(self):
        # Generate a buffer ID
        instance_vbo = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)

        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)  # Mark as per-instance attribute

        # Pass the instance positions data to the buffer
        glBufferData(GL_ARRAY_BUFFER, self.instance_positions.nbytes, self.instance_positions.flatten(), GL_STATIC_DRAW)
        # Enable and set up the attribute pointer for instance positions
        glEnableVertexAttribArray(1)  # Assuming 1 is the layout location for instancePos in your shader
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * self.instance_positions.itemsize, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)  # This makes it an instanced attribute
        # Unbind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return instance_vbo


    def setup_matrices(self):
        # Projection matrix (Perspective projection)
        projection = glm.perspective(glm.radians(45), self.window_width / self.window_height, 0.1, 100.0)

        # View matrix (Camera transformation)
        view = glm.lookAt(glm.vec3(20, 20, 30),  # Camera is at (2,2,3), in World Space
                               glm.vec3(0, 0, 0),  # Looks at the origin
                               glm.vec3(0, 1, 0))  # Head is up (set to 0,-1,0 to look upside-down)

        # Model matrix (Model transformation)
        model = glm.mat4(1.0)  # Initialize model matrix to identity matrix

        # Get the location of the uniforms from the shader
        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")

        return projection, view, model, model_loc, view_loc, proj_loc


    def render_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.window_width, self.window_height, GL_RGBA, GL_UNSIGNED_BYTE, self.volume_data)

        glBindVertexArray(self.vao)
        # Use your shader program
        glUseProgram(self.shader_program)
        # Now draw the quad
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

    def get_empty_volume(self):
        # Create an empty 3D numpy array for the volume data
        volume = np.zeros((self.simulation_height, self.simulation_width, self.simulation_length), dtype=np.uint8)

        # Randomly set some voxels to 255
        num_voxels_to_set = int(volume.size * 0.01)  # Set 1% of the voxels randomly for visualization
        indices = np.unravel_index(np.random.choice(volume.size, num_voxels_to_set, replace=False), volume.shape)
        volume[indices] = 1

        return volume

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

        # self.program.fade_image(self.cl_queue, (self.window_width, self.window_height), None, self.image_buf, self.settings_buffer, np.uint32(decay_amount))
        # self.program.draw_spores(self.cl_queue, (self.spore_count,), None, self.image_buf, self.spores_buffer, self.settings_buffer)
        # self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.image_buf, self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        # cl.enqueue_copy(self.cl_queue, self.volume_data, self.image_buf).wait()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(self.model))
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, glm.value_ptr(self.view))
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, glm.value_ptr(self.projection))

        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, len(self.instance_positions))
        glBindVertexArray(0)

        glUseProgram(0)


    def render_gui(self):
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
    game = Simulation3D(500, 500, 1000, target_framerate=100)
    game.run()
