import pyopencl as cl
import numpy as np
from OpenGL.GL import *
from Submit.game_engine import GameEngine


class ConwaysGameOfLife(GameEngine):
    def __init__(self, width, height, target_framerate=60, title='Conway\'s Game of Life', cl_file="Shaders/game_of_life.cl"):
        super().__init__(width, height, title, cl_file, target_framerate)
        self.image_data = self.initialize_pixel_grid(self.window_width, self.window_height)
        self.image_buf, self.new_image_buf = self.initialize_buffers(self.image_data)
        self.texture = self.initialize_texture()


    def initialize_pixel_grid(self, width, height):
        binary_grid = np.random.randint(2, size=(height, width), dtype=np.uint8)
        pixel_grid = np.empty((height, width, 4), dtype=np.uint8)
        pixel_grid[..., 0:3] = np.expand_dims(binary_grid, axis=-1) * 255
        pixel_grid[..., 3] = 255
        return pixel_grid

    def initialize_texture(self):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_data.shape[1], self.image_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)
        return texture

    def render_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.window_width, self.window_height, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def render(self):
        self.render_texture()

    def post_render(self):
        self.image_buf, self.new_image_buf = self.new_image_buf, self.image_buf

    def update(self):
        self.program.game_of_life(self.cl_queue, (self.window_width, self.window_height), None, self.image_buf, self.new_image_buf, np.uint32(self.window_width), np.uint32(self.window_height))
        cl.enqueue_copy(self.cl_queue, self.image_data, self.new_image_buf).wait()

    def render_gui(self):
        pass

if __name__ == '__main__':
    game = ConwaysGameOfLife(700, 700)
    game.run()