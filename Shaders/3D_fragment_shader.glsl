#version 330 core
out vec4 FragColor;

in vec3 FragPos; // Received from vertex shader

uniform float simulationWidth;
uniform float simulationHeight;
uniform float simulationLength;

void main()
{
    // Normalize position to [0, 1] for coloring
    vec3 normalizedPos = FragPos / vec3(simulationWidth, simulationHeight, simulationLength);

    // Define a base color so there's no true black
    vec3 baseColor = vec3(0.1, 0.3, 0.3); // Soft gray as an example

    // Ensure the final color is never completely black by mixing in the base color
    // Mix ratio can be adjusted as needed
    vec3 color = mix(baseColor, normalizedPos, 0.9);

    // Use the calculated color for the fragment color
    FragColor = vec4(color, 1.0);
}
