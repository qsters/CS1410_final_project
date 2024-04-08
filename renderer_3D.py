from OpenGL.GL import *
import numpy as np

class Renderer3D:
    def __init__(self, shader_program, instance_positions, instance_sizes, vertices, indices):
        self.shader_program = shader_program
        self.position_instance_vbo, self.size_instance_vbo = self.get_instance_buffers(instance_positions, instance_sizes)

        self.vao, self.vbo, self.ebo = self.setup_rendering(vertices, indices)

        self.uniform_locations = {}

        glEnable(GL_DEPTH_TEST)  # Enable depth test
        # Setup for rendering, such as enabling depth testing

    def add_uniform_location(self, name):
        self.uniform_locations[name] = glGetUniformLocation(self.shader_program, name)


    def get_instance_buffers(self, instance_positions, instance_sizes):
        # Generate and bind the buffer for instance positions
        instance_positions_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_positions_vbo)
        glBufferData(GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions.flatten(), GL_STATIC_DRAW)

        # Generate and bind the buffer for instance sizes
        instance_sizes_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, instance_sizes_vbo)
        glBufferData(GL_ARRAY_BUFFER, instance_sizes.nbytes, instance_sizes.flatten(), GL_STATIC_DRAW)

        # Unbind the GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return instance_positions_vbo, instance_sizes_vbo
    def setup_rendering(self, vertices, indices):
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
    # Implementation similar to your current setup_rendering, but abstracted
    # to be reusable for different types of objects and shaders

    def draw(self, num_instances):
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
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None, num_instances)

        # Clean up
        glBindVertexArray(0)
        glUseProgram(0)

        # Don't forget to unbind the GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_instance_data(self, new_instance_positions, new_instance_sizes):
        # Update instance positions
        glBindBuffer(GL_ARRAY_BUFFER, self.position_instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, new_instance_positions.nbytes, new_instance_positions.flatten(), GL_STATIC_DRAW)

        # Update instance sizes
        glBindBuffer(GL_ARRAY_BUFFER, self.size_instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, new_instance_sizes.nbytes, new_instance_sizes.flatten(), GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)  # Unbind the buffer