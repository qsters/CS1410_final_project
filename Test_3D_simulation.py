import math

import glfw
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine
import glm

class TestSimulation3D(GameEngine):
    def __init__(self, window_width, window_height, simulation_width=10, simulation_height=10, simulation_length=10, spore_count=300, title="Slime Mold Sim 2D", target_framerate=60):
        super().__init__(window_width, window_height, title, "Shaders/2d_simulation.cl", target_framerate)

        self.simulation_width = simulation_width
        self.simulation_height = simulation_height
        self.simulation_length = simulation_length

        self.fragment_shader = self.load_file("Shaders/3D_fragment_shader.glsl")
        self.vertex_shader = self.load_file("Shaders/3D_vertex_shader.glsl")

        self.volume_data = self.get_empty_volume()
        self.instance_positions, self.instance_sizes = self.get_instance_positions_and_sizes()
        self.position_instance_vbo, self.size_instance_vbo = self.get_instance_buffers()
        self.image_buf, _ = self.initialize_buffers(self.volume_data)

        self.shader_program = self.create_shader_program()
        self.vao, self.vbo, self.ebo = self.setup_rendering()

        self.uniform_loc_simulationWidth = glGetUniformLocation(self.shader_program, "simulationWidth")
        self.uniform_loc_simulationHeight = glGetUniformLocation(self.shader_program, "simulationHeight")
        self.uniform_loc_simulationLength = glGetUniformLocation(self.shader_program, "simulationLength")


        self.simulation_center = glm.vec3(
            self.simulation_width / 2,
            self.simulation_height / 2,
            self.simulation_length / 2
        )
        # Model matrix (Model transformation)
        self.model = glm.mat4(1.0)  # Initialize model matrix to identity matrix

        # Get the location of the uniforms from the shader
        self.model_loc = glGetUniformLocation(self.shader_program, "model")
        self.view_loc = glGetUniformLocation(self.shader_program, "view")
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")

        max_dimension = max(self.simulation_width, self.simulation_height, self.simulation_length)
        self.camera_distance = max_dimension * 2.5  # Adjust multiplier as needed for best view

        # Projection matrix (Perspective projection)
        self.projection = glm.perspective(glm.radians(45), self.window_width / self.window_height, 0.1, self.camera_distance * 2)

        self.camera_yaw = 45.0
        self.camera_pitch = 45.0

        self.view = self.get_view()

        if self.window:
            glfw.set_key_callback(self.window, self.key_callback)

        self.camera_speed = 15
        self.camera_delta = glm.vec2(0,0)

        glEnable(GL_DEPTH_TEST)  # Enable depth test

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


    def get_empty_volume(self):
        volume = np.zeros((self.simulation_height, self.simulation_width, self.simulation_length), dtype=np.float32)
        return volume


    def get_instance_positions_and_sizes(self):
        # Assuming volume_data is a NumPy array of shape (height, width, length)
        # Find indices where value > 0
        nonzero_indices = np.argwhere(self.volume_data > 0)

        # Normalize sizes based on voxel values
        sizes = self.volume_data[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

        # Change type of data to floats
        positions = nonzero_indices.astype(np.float32)
        sizes = sizes.astype(np.float32)
        print(positions)
        return positions, sizes

    def get_view(self):
        # Recalculate the camera front vector
        front = glm.vec3(
            math.cos(glm.radians(self.camera_yaw)) * math.cos(glm.radians(self.camera_pitch)),
            math.sin(glm.radians(self.camera_pitch)),
            math.sin(glm.radians(self.camera_yaw)) * math.cos(glm.radians(self.camera_pitch))
        )
        front = glm.normalize(front)

        # Now update the view matrix
        camera_position = self.simulation_center + front * self.camera_distance
        view = glm.lookAt(camera_position, self.simulation_center, glm.vec3(0, 1, 0))
        return view

    def key_callback(self, window, key, scancode, action, mods):
        # Adjust the camera's yaw and pitch based on arrow key input
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_UP:
                self.camera_delta[1] = 1
            if key == glfw.KEY_DOWN:
                self.camera_delta[1] = -1
            if key == glfw.KEY_RIGHT:
                self.camera_delta[0] = 1
            if key == glfw.KEY_LEFT:
                self.camera_delta[0] = -1
        else:
            self.camera_delta *= 0



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

        # Cube vertices
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Cube indices
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Instance positions setup should be here, within VAO setup
        glBindBuffer(GL_ARRAY_BUFFER, self.position_instance_vbo)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        # Unbind VAO (not the EBO or instance VBO)
        glBindVertexArray(0)
        # It's okay to unbind the GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return vao, vbo, ebo


    def get_instance_buffers(self):

        # Generate and bind the buffer for instance positions
        instance_positions_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_positions_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.instance_positions.nbytes, self.instance_positions.flatten(), GL_STATIC_DRAW)

        # Generate and bind the buffer for instance sizes
        instance_sizes_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_sizes_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.instance_sizes.nbytes, self.instance_sizes.flatten(), GL_STATIC_DRAW)

        # Unbind the GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return instance_positions_vbo, instance_sizes_vbo


    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        # Set simulation sizes
        glUniform1f(self.uniform_loc_simulationWidth, float(self.simulation_width))
        glUniform1f(self.uniform_loc_simulationHeight, float(self.simulation_height))
        glUniform1f(self.uniform_loc_simulationLength, float(self.simulation_length))

        # Update matrices uniforms
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(self.model))
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, glm.value_ptr(self.view))
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, glm.value_ptr(self.projection))

        # Bind VAO
        glBindVertexArray(self.vao)


        # Bind the instance positions buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.position_instance_vbo)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)  # This makes it an instanced attribute for position

        # Bind the instance sizes buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.size_instance_vbo)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)  # This makes it an instanced attribute for size

        # Draw the instances
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, len(self.instance_positions))

        # Clean up
        glBindVertexArray(0)
        glUseProgram(0)

        # Don't forget to unbind the GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def update(self):
        if self.camera_delta == [0, 0]:  # return if zero
            return

        self.camera_delta = glm.normalize(self.camera_delta)  # Normalize the delta vector

        # Use the normalized values for pitch and yaw adjustments
        self.camera_pitch += self.camera_delta.y * self.delta_time * self.camera_speed
        self.camera_yaw -= self.camera_delta.x * self.delta_time * self.camera_speed

        self.view = self.get_view()


if __name__ == '__main__':
    game = TestSimulation3D(500, 500, target_framerate=45)
    game.run()
