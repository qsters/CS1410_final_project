from OpenGL.GL import *


class ShaderProgram:
    def __init__(self, vertex_file_path, fragment_file_path):
        self.vertex_shader_source = self.load_file(vertex_file_path)
        self.fragment_shader_source = self.load_file(fragment_file_path)
        self.program = self.create_shader_program()

    @staticmethod
    def load_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode('utf-8')
            raise RuntimeError("Shader compilation error: " + error)
        return shader


    def create_shader_program(self):
        vertex_shader = self.compile_shader(self.vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.fragment_shader_source, GL_FRAGMENT_SHADER)

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

# Implementation here
