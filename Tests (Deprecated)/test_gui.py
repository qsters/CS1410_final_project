import imgui
from Submit.game_engine import GameEngine  # Assuming GameEngine is in game_engine.py

class GameEngineWithGUI(GameEngine):
    def __init__(self, width, height, title, cl_file="Shaders/2d_simulation.cl", target_framerate=60):
        super().__init__(width, height, title, cl_file, target_framerate)
        # ImGui context is already created in the GameEngine.__init__
        self.string = ""
        self.f = 0.5

    def render_gui(self):
        if imgui.begin("Custom window", True):
            imgui.text("Hello, world!")

            if imgui.button("OK"):
                print(f"String: {self.string}")
                print(f"Float: {self.f}")

            _, self.string = imgui.input_text("A String", self.string, 256)
            _, self.f = imgui.slider_float("float", self.f, 0.25, 1.5)

            imgui.end()

    def render(self):
        pass

    def update(self):
        pass

if __name__ == "__main__":
    game = GameEngineWithGUI(1280, 720, "Game Engine with GUI Test")
    game.run()
