import imgui
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine
import glm
import pyopencl as cl

from shader_program import ShaderProgram
from simulation_renderer_3D import SimulationRenderer3D
from camera_mover import CameraHandler3D


class Simulation3D(GameEngine):
    def __init__(self, window_width, window_height, simulation_size=10, spore_count=300, title="Slime Mold Sim 2D",
                 target_framerate=60):
        super().__init__(window_width, window_height, title, "Shaders/3d_simulation.cl", target_framerate)
        # Set basic values
        self.simulation_size = simulation_size

        self.spore_count = spore_count
        self.spores = self.initialize_spores()
        self.spores_buffer = self.initialize_buffer(self.spores)

        self.spore_speed = 17
        self.decay_speed = 0.4
        self.sensor_distance = 14
        self.turn_speed = 11

        # Dtype for the settings
        self.settings_dtype = np.dtype([
            ('spore_count', np.uint32),
            ('simulation_size', np.uint32),
            ('spore_speed', np.float32),
            ('decay_speed', np.float32),
            ('turn_speed', np.float32),
            ('sensor_distance', np.float32),
        ])

        # Buffers

        # Settings Buffer
        self.settings = np.array([(self.spore_count, self.simulation_size, self.spore_speed, self.decay_speed,
                                   self.turn_speed, self.sensor_distance)], dtype=self.settings_dtype)
        self.settings_buffer = self.initialize_buffer(self.settings)

        # Random Seeds buffer
        self.random_seeds = np.random.randint(0, 2 ** 32 - 1, size=self.spore_count + 1, dtype=np.uint32)
        self.random_seeds_buffer = self.initialize_buffer(self.random_seeds)

        # Volume Buffer
        self.volume_data = self.get_empty_volume()
        self.volume_buffer = self.initialize_buffer(self.volume_data)

        # Setup Shader program
        self.shader_program = ShaderProgram("Shaders/3D_vertex_shader.glsl", "Shaders/3D_fragment_shader.glsl")


        # Get instance data
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

        # Setup renderer
        self.renderer = SimulationRenderer3D(self.shader_program.program, self.instance_positions, self.instance_sizes, vertices,
                                             indices)

        # Add the uniforms for the shaders
        self.renderer.add_uniform_location("simulationSize")
        self.renderer.add_uniform_location("model")
        self.renderer.add_uniform_location("view")
        self.renderer.add_uniform_location("projection")

        # Setup camera stuff
        simulation_center = glm.vec3(
            self.simulation_size / 2,
            self.simulation_size / 2,
            self.simulation_size / 2
        )

        camera_multiplier = 2.5  # Adjust multiplier as needed for best view
        camera_distance = self.simulation_size * camera_multiplier

        camera_speed = 3

        # Model matrix (Model transformation)
        self.model = glm.mat4(1.0)  # Initialize model matrix to identity matrix

        # Projection matrix (Perspective projection)
        self.projection = glm.perspective(glm.radians(45), self.window_width / self.window_height, 0.1,
                                          camera_distance * 2)

        self.camera_mover = CameraHandler3D(45.0, 45.0, simulation_center, camera_distance, camera_speed, self.window)

    def get_empty_volume(self):
        """Generates empty volume by the simulation size"""
        volume = np.zeros((self.simulation_size, self.simulation_size, self.simulation_size), dtype=np.float32)
        return volume

    def initialize_spores(self):
        """Creates spores with random values"""
        # Define the structured datatype for a 2D spore
        spore_dtype = np.dtype([
            ('x', np.float32),  # Position Vector
            ('y', np.float32),
            ('z', np.float32),
            ('pad', np.float32),  # Padding for data
            ('dir_x', np.float32),  # Directino Vector
            ('dir_y', np.float32),
            ('dir_z', np.float32),
            ('pad2', np.float32),  # Padding for data
        ])

        # Initialize empty array of spores
        spores = np.zeros(self.spore_count, dtype=spore_dtype)

        # Randomize positions within the bounds of the simulation size
        spores['x'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)
        spores['y'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)
        spores['z'] = np.random.uniform(0, self.simulation_size, size=self.spore_count)

        # Generate random direction vectors and normalize them
        directions = np.random.randn(self.spore_count, 3)  # Generate random directions
        norms = np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Calculate norms
        normalized_directions = directions / norms  # Normalize

        # Assign normalized directions to spores
        spores['dir_x'] = normalized_directions[:, 0]
        spores['dir_y'] = normalized_directions[:, 1]
        spores['dir_z'] = normalized_directions[:, 2]

        return spores

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
        """Runs Kernels, updates data, and camera position"""

        self.program.decay_trails(self.cl_queue, (self.simulation_size, self.simulation_size, self.simulation_size),
                                  None, self.volume_buffer, self.settings_buffer, np.float32(self.delta_time))

        self.program.draw_spores(self.cl_queue, (self.spore_count,), None,
                                 self.volume_buffer, self.spores_buffer, self.settings_buffer)

        self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.volume_buffer,
                                 self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        self.cl_queue.finish()

        cl.enqueue_copy(self.cl_queue, self.volume_data, self.volume_buffer).wait()

        self.instance_positions, self.instance_sizes = self.get_instance_positions_and_sizes()
        self.renderer.update_instance_data(self.instance_positions, self.instance_sizes)
        self.camera_mover.update_view(self.delta_time)

    def render_gui(self):
        # Set the window's background alpha (transparency) to 0.7 (1.0 is opaque, 0.0 is transparent)
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.8)

        # Start a new ImGui window
        if imgui.begin("Simulation Parameters"):

            # Slider for spore_speed
            changed, self.spore_speed = imgui.slider_float("Spore Speed", self.spore_speed, 0.1, 30.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for decay_speed
            changed, self.decay_speed = imgui.slider_float("Decay Speed", self.decay_speed, 0.01, 2.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for sensor_distance
            changed, self.sensor_distance = imgui.slider_float("Sensor Distance", self.sensor_distance, 1.0, 20.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

            # Slider for turn_speed
            changed, self.turn_speed = imgui.slider_float("Turn Speed", self.turn_speed, 0.0, 25.0)
            if changed:
                # Update the settings buffer if necessary
                self.update_settings_buffer()

        imgui.end()
        imgui.pop_style_var()

    def update_settings_buffer(self):
        # Create a new settings array
        settings = np.array([(self.spore_count, self.simulation_size,
                              self.spore_speed, self.decay_speed, self.turn_speed,
                              self.sensor_distance)], dtype=self.settings_dtype)

        # Update the buffer with the new settings
        cl.enqueue_copy(self.cl_queue, self.settings_buffer, settings)

if __name__ == '__main__':
    game = Simulation3D(500, 500, 50,  target_framerate=30, spore_count=1000)
    game.run()
