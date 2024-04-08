#version 330 core
layout (location = 0) in vec3 aPos; // Cube vertex positions
layout (location = 1) in vec3 instancePos; // Instance positions
layout (location = 2) in float instanceSize; // New attribute for size

out vec3 FragPos; // Pass position to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Function to create a scaling matrix
mat4 scale(float scaleFactor) {
    return mat4(
        scaleFactor, 0.0, 0.0, 0.0,
        0.0, scaleFactor, 0.0, 0.0,
        0.0, 0.0, scaleFactor, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

// Function to create a translation matrix
mat4 translate(vec3 trans) {
    return mat4(
        1.0, 0.0, 0.0, trans.x,
        0.0, 1.0, 0.0, trans.y,
        0.0, 0.0, 1.0, trans.z,
        0.0, 0.0, 0.0, 1.0
    );
}

void main() {
    // Manually create the scaled and translated model matrix
    mat4 scaledModel = model * scale(1);
    mat4 translatedModel = translate(instancePos) * scaledModel;
    gl_Position = projection * view * translatedModel * vec4(aPos, 1.0);
    FragPos = instancePos; // Pass instance position to fragment shader
}
