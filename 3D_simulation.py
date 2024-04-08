import math

import glfw
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine
import glm
import pyopencl as cl

from shader_program import ShaderProgram
from renderer_3D import Renderer3D
from camera_mover import CameraMover


class Simulation3D(GameEngine):
    def __init__(self, window_width, window_height, simulation_size=10, spore_count=300, title="Slime Mold Sim 2D",
                 target_framerate=60):
        super().__init__(window_width, window_height, title, "Shaders/3d_simulation.cl", target_framerate)

        self.simulation_size = simulation_size

        self.spore_count = spore_count
        self.spores = self.initialize_spores()
        self.spores_buffer = self.initialize_buffer(self.spores)

        self.spore_speed = 100
        self.decay_speed = 1
        self.sensor_distance = 5
        self.turn_speed = 3

        self.settings = self.create_settings()
        self.settings_buffer = self.initialize_buffer(self.settings)

        self.random_seeds = np.random.randint(0, 2 ** 32 - 1, size=self.spore_count, dtype=np.uint32)
        self.random_seeds_buffer = self.initialize_buffer(self.random_seeds)

        self.shader_program = ShaderProgram("Shaders/3D_vertex_shader.glsl", "Shaders/3D_fragment_shader.glsl")

        self.volume_data = self.get_empty_volume()
        self.volume_buffer = self.initialize_buffer(self.volume_data)

        self.instance_positions, self.instance_sizes = self.get_instance_positions_and_sizes()

        # Vertex data: positions and texture coordinates for a fullscreen quad
        # Define the 8 vertices of the cube
        vertices = np.array([
            -0.5, -0.5, -0.5,  # Vertex 0
            0.5, -0.5, -0.5,  # Vertex 1
            0.5, 0.5, -0.5,  # Vertex 2
            -0.5, 0.5, -0.5,  # Vertex 3
            -0.5, -0.5, 0.5,  # Vertex 4
            0.5, -0.5, 0.5,  # Vertex 5
            0.5, 0.5, 0.5,  # Vertex 6
            -0.5, 0.5, 0.5  # Vertex 7
        ], dtype=np.float32)

        # Define the 12 triangles (2 per face) that make up the cube
        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front face
            1, 5, 6, 6, 2, 1,  # Right face
            7, 6, 5, 5, 4, 7,  # Back face
            4, 0, 3, 3, 7, 4,  # Left face
            4, 5, 1, 1, 0, 4,  # Bottom face
            3, 2, 6, 6, 7, 3  # Top face
        ], dtype=np.uint32)

        self.renderer = Renderer3D(self.shader_program.program, self.instance_positions, self.instance_sizes, vertices,
                                   indices)

        # Add the uniforms for the shaders
        self.renderer.add_uniform_location("simulationSize")
        self.renderer.add_uniform_location("model")
        self.renderer.add_uniform_location("view")
        self.renderer.add_uniform_location("projection")

        simulation_center = glm.vec3(
            self.simulation_size / 2,
            self.simulation_size / 2,
            self.simulation_size / 2
        )

        camera_multiplier = 2.5
        camera_distance = self.simulation_size * camera_multiplier  # Adjust multiplier as needed for best view

        camera_speed = 3

        # Model matrix (Model transformation)
        self.model = glm.mat4(1.0)  # Initialize model matrix to identity matrix

        # Projection matrix (Perspective projection)
        self.projection = glm.perspective(glm.radians(45), self.window_width / self.window_height, 0.1,
                                          camera_distance * 2)

        self.camera_mover = CameraMover(45.0, 45.0, simulation_center, camera_distance, camera_speed, self.window)

    def get_empty_volume(self):
        volume = np.zeros((self.simulation_size, self.simulation_size, self.simulation_size), dtype=np.float32)
        return volume

    def initialize_spores(self):
        # Define the structured datatype for a 2D spore
        spore_dtype = np.dtype([
            ('x', np.float32),  # x position
            ('y', np.float32),  # y position
            ('z', np.float32),  # z position
            ('angle', np.float32),  # azimuth angle
            ('inclination', np.float32)  # azimuth angle
        ])

        # Initialize empty array of spores
        spores = np.zeros(self.spore_count, dtype=spore_dtype)

        # Randomize positions within the bounds of height and width
        spores['x'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)
        spores['y'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)
        spores['z'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)

        spores['angle'] = np.random.uniform(0, 2 * math.pi, size=self.spore_count)
        spores['inclination'] = np.random.uniform(0, 2 * math.pi, size=self.spore_count)

        return spores

    def create_settings(self):
        settings_dtype = np.dtype([
            ('spore_count', np.uint32),
            ('simulation_size', np.uint32),
            ('spore_speed', np.float32),
            ('decay_speed', np.float32),
            ('turn_speed', np.float32),
            ('sensor_distance', np.float32),
        ])

        settings = np.array([(self.spore_count, self.simulation_size, self.spore_speed, self.decay_speed,
                              self.turn_speed, self.sensor_distance)], dtype=settings_dtype)
        return settings

    def get_instance_positions_and_sizes(self):
        # Find indices where value > 0
        nonzero_indices = np.argwhere(self.volume_data > 0)

        # Normalize sizes based on voxel values
        sizes = self.volume_data[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

        # Change type of data to floats
        positions = nonzero_indices.astype(np.float32)
        sizes = sizes.astype(np.float32)
        return positions, sizes

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program.program)

        # Set simulation sizes
        glUniform1f(self.renderer.uniform_locations["simulationSize"], float(self.simulation_size))

        # Update matrices uniforms
        glUniformMatrix4fv(self.renderer.uniform_locations["model"], 1, GL_FALSE, glm.value_ptr(self.model))
        glUniformMatrix4fv(self.renderer.uniform_locations["view"], 1, GL_FALSE, glm.value_ptr(self.camera_mover.view))
        glUniformMatrix4fv(self.renderer.uniform_locations["projection"], 1, GL_FALSE, glm.value_ptr(self.projection))

        self.renderer.draw(len(self.instance_positions))

    def update(self):
        self.program.decay_trails(self.cl_queue, (self.simulation_size, self.simulation_size, self.simulation_size), None, self.volume_buffer, self.settings_buffer, np.float32(self.delta_time))

        self.program.draw_spores(self.cl_queue, (self.spore_count,), None, self.volume_buffer, self.spores_buffer,
                                 self.settings_buffer)

        self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.volume_buffer, self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        cl.enqueue_copy(self.cl_queue, self.volume_data, self.volume_buffer).wait()

        self.instance_positions, self.instance_sizes = self.get_instance_positions_and_sizes()
        self.renderer.update_instance_data(self.instance_positions, self.instance_sizes)
        self.camera_mover.update_view(self.delta_time)


if __name__ == '__main__':
    game = Simulation3D(500, 500, target_framerate=10, spore_count=1)
    game.run()
