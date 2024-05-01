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

        # self.spore_speed = 5.5
        # self.decay_speed = 0.4
        # self.sensor_distance = 4.6
        # self.turn_speed = 7
        self.spore_speed = 3
        self.decay_speed = 0.4
        self.sensor_distance = 4.6
        self.turn_speed = 0

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

        spores['x'][0] = 25
        spores['y'][0] = 25
        spores['z'][0] = 25

        spores['dir_x'][0] = 0
        spores['dir_y'][0] = 0
        spores['dir_z'][0] = 1


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

        self.decay_spores()
        self.draw_spores()
        self.move_spores()
        self.sense_and_turn(debug=True)

        self.volume_buffer = self.initialize_buffer(self.volume_data)


        # self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.volume_buffer,
        #                          self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        # self.cl_queue.finish()

        # cl.enqueue_copy(self.cl_queue, self.volume_data, self.volume_buffer).wait()
        # cl.enqueue_copy(self.cl_queue, self.spores, self.spores_buffer).wait()

        self.instance_positions, self.instance_sizes = self.get_instance_positions_and_sizes()
        self.renderer.update_instance_data(self.instance_positions, self.instance_sizes)
        self.camera_mover.update_view(self.delta_time)

    def decay_spores(self):
        self.volume_data -= self.delta_time * self.decay_speed
        self.volume_data = np.clip(self.volume_data, 0,1)

    def draw_spores(self):
        simulation_size = self.settings['simulation_size']

        # Extract positions
        x = self.spores['x'].astype(int)
        y = self.spores['y'].astype(int)
        z = self.spores['z'].astype(int)

        # Clamp values to ensure they are within the bounds of the volume
        x = np.clip(x, 0, simulation_size - 1)
        y = np.clip(y, 0, simulation_size - 1)
        z = np.clip(z, 0, simulation_size - 1)

        # Update the volume data at the clamped coordinates
        self.volume_data[x, y, z] = 1

    def move_spores(self):
        """Move spores in their respective directions by the current spore speed."""
        # Calculate the change in position for each spore based on their direction and speed
        delta_x = self.spores['dir_x'] * self.spore_speed * self.delta_time
        delta_y = self.spores['dir_y'] * self.spore_speed * self.delta_time
        delta_z = self.spores['dir_z'] * self.spore_speed * self.delta_time

        # Update spore positions
        self.spores['x'] += delta_x
        self.spores['y'] += delta_y
        self.spores['z'] += delta_z

        # Wrap positions around the simulation boundaries to create a toroidal space
        self.spores['x'] %= self.simulation_size
        self.spores['y'] %= self.simulation_size
        self.spores['z'] %= self.simulation_size


    def sense_and_turn(self, debug=False):
        angle_offset = np.radians(45)
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)

        sense_directions = np.array([
            [1, 0, 0],  # Forward
            [cos_offset, sin_offset, 0],  # Left-forward
            [cos_offset, -sin_offset, 0],  # Right-forward
            [cos_offset, 0, sin_offset],  # Up-forward (tilting up)
            [cos_offset, 0, -sin_offset],  # Down-forward (tilting down)
        ])

        for idx in range(self.spore_count):
            dir_vector = np.array([self.spores['dir_x'][idx], self.spores['dir_y'][idx], self.spores['dir_z'][idx]])
            forward = glm.normalize(glm.vec3(dir_vector))

            # Assuming Y is global up, but Z is forward
            global_up = glm.vec3(0, 1, 0)
            right = glm.normalize(glm.cross(global_up, forward))
            up = glm.normalize(glm.cross(forward, right))

            rotation_matrix = np.array([right, up, forward]).T

            if debug:
                print("Rotation Matrix:\n", rotation_matrix)
                print("forward: ", forward)
                print("right: ", right)
                print("up: ", up)

            concentration = np.zeros(5)
            position = np.array([self.spores['x'][idx], self.spores['y'][idx], self.spores['z'][idx]])
            for i, sense_dir in enumerate(sense_directions):
                rotated_sense_dir = rotation_matrix.dot(sense_dir)
                sense_point = (position + rotated_sense_dir * self.sensor_distance) % self.simulation_size
                concentration[i] = self.volume_data[tuple(sense_point.astype(int))]
                if debug:
                    print(f"Original: {sense_dir}, Rotated: {rotated_sense_dir}")
                    debug_point = tuple(sense_point.astype(int))
                    self.volume_data[debug_point] = 1

            best_dir_idx = np.argmax(concentration)
            best_direction = rotation_matrix.dot(sense_directions[best_dir_idx])
            turn_rate = self.turn_speed * self.delta_time
            new_dir = glm.mix(glm.vec3(dir_vector), glm.vec3(best_direction), turn_rate)
            new_dir = glm.normalize(new_dir)
            self.spores['dir_x'][idx], self.spores['dir_y'][idx], self.spores['dir_z'][idx] = new_dir.x, new_dir.y, new_dir.z




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
    game = Simulation3D(1000, 1000, 50,  target_framerate=30, spore_count=1)
    game.run()