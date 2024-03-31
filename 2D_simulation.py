import math

import pyopencl as cl
import numpy as np
from OpenGL.GL import *
from game_engine import GameEngine


class Simulation2D(GameEngine):
    def __init__(self, width, height, spore_count=300, title="Slime Mold Sim 2D", cl_file="Shaders/2d_simulation.cl", target_framerate=60):
        super().__init__(width, height, title, cl_file, target_framerate)

        self.image_data = self.get_empty_image()
        self.image_buf, _ = self.initialize_buffers(self.image_data)
        self.texture = self.initialize_texture()

        self.spore_count = spore_count
        self.spores = self.initialize_spores()
        self.spores_buffer = self.initialize_buffer(self.spores)

        self.random_seeds = np.random.randint(0, 2**32 - 1, size=self.spore_count, dtype=np.uint32)
        self.random_seeds_buffer = self.initialize_buffer(self.random_seeds)

        self.spore_speed = 100
        self.decay_speed = .1
        self.sensor_distance = 5

        self.decay_accumulator = 0

        self.settings = self.create_settings()
        self.settings_buffer = self.initialize_buffer(self.settings)

    def create_settings(self):
        settings_dtype = np.dtype([
            ('spore_count', np.uint32),
            ('screen_height', np.uint32),
            ('screen_width', np.uint32),
            ('spore_speed', np.float32),
            ('decay_speed', np.float32),
            ('sensor_distance', np.float32)
        ])

        settings = np.array([(self.spore_count, self.height, self.width, self.spore_speed, self.decay_speed, self.sensor_distance)], dtype=settings_dtype)
        return settings

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

    def get_empty_image(self):
        pixel_grid = np.empty((self.height, self.width, 4), dtype=np.uint8)
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
        spores['x'] = np.random.randint(0, self.width, size=self.spore_count)
        spores['y'] = np.random.randint(0, self.height, size=self.spore_count)
        spores['angle'] = np.random.uniform(0, 2 * math.pi, size=self.spore_count)

        return spores

    def update(self):
        # Calculate the total decay amount to add to the accumulator
        self.decay_accumulator += self.decay_speed * self.delta_time * 255

        # Extract the whole number part of the decay amount to apply
        decay_amount = int(self.decay_accumulator)

        # Update the accumulator to retain only the fractional part
        self.decay_accumulator -= decay_amount

        self.program.fade_image(self.cl_queue, (self.width, self.height), None, self.image_buf, self.settings_buffer, np.uint32(decay_amount))
        self.program.draw_spores(self.cl_queue, (self.spore_count,), None, self.image_buf, self.spores_buffer, self.settings_buffer)
        self.program.move_spores(self.cl_queue, (self.spore_count,), None, self.spores_buffer, self.image_buf, self.random_seeds_buffer, self.settings_buffer, np.float32(self.delta_time))

        cl.enqueue_copy(self.cl_queue, self.image_data, self.image_buf).wait()

    def render(self):
        self.render_texture()

if __name__ == '__main__':
    game = Simulation2D(500, 500, 1000)
    game.run()
