#version 330 core
out vec4 FragColor;

in vec3 FragPos; // Received from vertex shader

uniform float simulationSize;

void main()
{
    // Normalize position to [0, 1] for coloring
    vec3 normalizedPos = FragPos / vec3(simulationSize, simulationSize, simulationSize);

    // Define a base color so there's no true black
    vec3 baseColor = vec3(0.8, 0.8, 0.8); // Soft gray as an example

    // Ensure the final color is never completely black by mixing in the base color
    // Mix ratio can be adjusted as needed
    vec3 color = mix(baseColor, normalizedPos, 0.9);

    // Use the calculated color for the fragment color
    FragColor = vec4(color, 1.0);
}
