import glm
import math
import glfw

class CameraHandler3D:
    def __init__(self, camera_yaw, camera_pitch, look_at, camera_distance, camera_speed, window):
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch

        self.camera_speed = 15
        self.camera_delta = glm.vec2(0,0)

        self.look_at = look_at
        self.camera_distance = camera_distance

        self.view = self.get_current_view()

        if window:
            glfw.set_key_callback(window, self.key_callback)

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

    def get_current_view(self):
        # Recalculate the camera front vector
        front = glm.vec3(
            math.cos(glm.radians(self.camera_yaw)) * math.cos(glm.radians(self.camera_pitch)),
            math.sin(glm.radians(self.camera_pitch)),
            math.sin(glm.radians(self.camera_yaw)) * math.cos(glm.radians(self.camera_pitch))
        )
        front = glm.normalize(front)

        # Now update the view matrix
        camera_position = self.look_at + front * self.camera_distance
        view = glm.lookAt(camera_position, self.look_at, glm.vec3(0, 1, 0))
        return view

    def update_view(self, delta_time):
        if self.camera_delta == [0, 0]:  # return if zero
            return

        self.camera_delta = glm.normalize(self.camera_delta) * self.camera_speed

        # Use the normalized values for pitch and yaw adjustments
        self.camera_pitch += self.camera_delta.y * delta_time * self.camera_speed
        self.camera_yaw -= self.camera_delta.x * delta_time * self.camera_speed

        # # Clamp camera pitch to prevent 'flipping'
        pitch_limit = 89.0
        self.camera_pitch = max(-pitch_limit, min(self.camera_pitch, pitch_limit))

        self.view = self.get_current_view()