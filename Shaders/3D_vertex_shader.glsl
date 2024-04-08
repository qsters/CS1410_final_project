#version 330 core
layout (location = 0) in vec3 aPos; // Cube vertex positions
layout (location = 1) in vec3 instancePos; // Instance positions
layout (location = 2) in float instanceSize; // Instance size

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;

void main() {
    // Scale matrix based on the instance size
    mat4 scaleMatrix = mat4(
        instanceSize, 0.0, 0.0, 0.0,
        0.0, instanceSize, 0.0, 0.0,
        0.0, 0.0, instanceSize, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    vec3 scaledPos = instancePos * (1 / instanceSize);

    // Apply scaling then translation to the model matrix
    mat4 scaledModel = model  * scaleMatrix;
    mat4 modelView = scaledModel * mat4(1.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        scaledPos.x, scaledPos.z, scaledPos.y, 1.0);

    gl_Position = projection * view * modelView * vec4(aPos, 1.0);
    FragPos = instancePos; // Pass scaled position to fragment shader
}
