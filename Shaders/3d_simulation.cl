#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


typedef struct {
    float3 position;  // Use float2 for position
    float3 direction;
} Spore;

typedef struct {
    uint spore_count;
    uint simulation_size;
    float spore_speed;
    float decay_speed;
    float turn_speed;
    float sensor_distance;
} Settings;

//// Function prototypes
uint hash(uint x);
float scaleToRange01(uint x);
float sense(__global Spore* spore, __global float* volume, __global const Settings* settings, float3 direction, float3 forward);
void draw_sensor(__global Spore* spore, __global float* volume, __global const Settings* settings, float3 direction, float3 forward);

float sense(__global Spore* spore, __global float* volume, __global const Settings* settings, float3 direction, float3 forward) {
        // Calculate the sampling position using the direction vector
        float3 averagePos = normalize(forward + direction);

        float3 samplePos = spore->position + averagePos * settings->sensor_distance;

        // Clamp the sampling position to be within the simulation bounds
        uint sampleX = clamp((uint)samplePos.x, 0u, settings->simulation_size - 1u);
        uint sampleY = clamp((uint)samplePos.y, 0u, settings->simulation_size - 1u);
        uint sampleZ = clamp((uint)samplePos.z, 0u, settings->simulation_size - 1u);

        // Calculate the linear index in the volume array
        uint idx = sampleZ * settings->simulation_size * settings->simulation_size + sampleY * settings->simulation_size + sampleX;

        // return value
        return volume[idx];
}

void draw_sensor(__global Spore* spore, __global float* volume, __global const Settings* settings, float3 direction, float3 forward) {
    // Calculate the sampling position using the direction vector
    float3 averagePos = normalize(forward + direction);

    float3 samplePos = spore->position + averagePos * settings->sensor_distance;

    // Clamp the sampling position to be within the simulation bounds
    uint sampleX = clamp((uint)samplePos.x, 0u, settings->simulation_size - 1u);
    uint sampleY = clamp((uint)samplePos.y, 0u, settings->simulation_size - 1u);
    uint sampleZ = clamp((uint)samplePos.z, 0u, settings->simulation_size - 1u);

    // Calculate the linear index in the volume array
    uint idx = sampleZ * settings->simulation_size * settings->simulation_size + sampleY * settings->simulation_size + sampleX;

    // Mark the sensor position in the volume
    volume[idx] = 0.5; // Assign a value to indicate a sensor's position
}


uint hash(uint x) {
    x += (x << 10);
    x ^= (x >> 6);
    x += (x << 3);
    x ^= (x >> 11);
    x += (x << 15);
    return x;
}

float scaleToRange01(uint x) {
    // Explicitly specify the divisor as a float to ensure floating-point division.
    return x / (float)0xFFFFFFFF;
}

__kernel void draw_spores(__global float* volume, __global Spore* spores, __global const Settings* settings) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    Spore s = spores[idx];
    uint x = (int)(s.position.x);
    uint y = (int)(s.position.y);
    uint z = (int)(s.position.z);

    // Ensure the coordinates are within the image bounds
    if (x < settings->simulation_size && y < settings->simulation_size && z < settings->simulation_size) {
        uint volume_idx = z * settings->simulation_size * settings->simulation_size + y * settings->simulation_size + x;
        volume[volume_idx] = 1.0f; // Place a 1 at the position of the spore
    }
}

__kernel void draw_sensors(__global float* volume, __global Spore* spores, __global const Settings* settings) {
    uint idx = get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    float3 globalUp = (float3)(0.0f, 0.0f, 1.0f); // Global up
    float3 sporeDirection = spores[idx].direction; // Assuming spores[].direction is already defined

    // Generate the local right vector
    float3 rightVector = cross(sporeDirection, globalUp);
    if (length(rightVector) == 0) { // Handle parallel or antiparallel direction
        // Fallback or adjust rightVector
        rightVector = (float3)(1.0f, 0.0f, 0.0f);
    }
    rightVector = normalize(rightVector);

    // Generate local up vector based on right and forward vectors
    float3 upVector = cross(rightVector, sporeDirection);
    upVector = normalize(upVector);

    draw_sensor(&spores[idx], volume, settings, sporeDirection, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, rightVector, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, -rightVector, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, upVector, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, -upVector, sporeDirection);
}


__kernel void move_spores(__global Spore* spores, __global float* volume, __global uint* random_seeds, __global const Settings* settings, const float delta_time) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    float3 globalUp = (float3)(0.0f, 0.0f, 1.0f); // Global up
    float3 sporeDirection = spores[idx].direction; // Assuming spores[].direction is already defined

    // Generate the local right vector
    float3 rightVector = cross(sporeDirection, globalUp);
    if (length(rightVector) == 0) { // Handle parallel or antiparallel direction
        // Fallback or adjust rightVector
        rightVector = (float3)(1.0f, 0.0f, 0.0f);
    }
    rightVector = normalize(rightVector);

    // Generate local up vector based on right and forward vectors
    float3 upVector = cross(rightVector, sporeDirection);
    upVector = normalize(upVector);

    float forwardWeight = sense(&spores[idx], volume, settings, sporeDirection, sporeDirection);
    float rightWeight = sense(&spores[idx], volume, settings, rightVector, sporeDirection);
    float leftWeight = sense(&spores[idx], volume, settings, -rightVector, sporeDirection);
    float upWeight = sense(&spores[idx], volume, settings, upVector, sporeDirection);
    float downWeight = sense(&spores[idx], volume, settings, -upVector, sporeDirection);

    float3 directionChange = (float3)(0,0,0);

    if (forwardWeight < rightWeight || forwardWeight < leftWeight) {
        // Decide turning based on left and right weights
        if (rightWeight > leftWeight) {
            directionChange += rightVector;
        } else if (leftWeight > rightWeight) {
            directionChange -= rightVector;
        }
    }

    if (forwardWeight < upWeight || forwardWeight < downWeight) {
        // Decide inclination based on up and down weights
        if (upWeight > downWeight) {
            directionChange += upVector;
        } else if (downWeight > upWeight) {
            directionChange -= upVector;
        }
    }

    float3 newDirection = sporeDirection + directionChange * settings->spore_speed * delta_time;
    newDirection = normalize(newDirection);

    float3 newPosition = spores[idx].position + newDirection * settings->spore_speed * delta_time;

    float3 storePosition = newPosition;

    // make sure things are in bounds
    newPosition.x = clamp(newPosition.x, 0.0f, (float)(settings->simulation_size - 1));
    newPosition.y = clamp(newPosition.y, 0.0f, (float)(settings->simulation_size - 1));
    newPosition.z = clamp(newPosition.z, 0.0f, (float)(settings->simulation_size - 1));

    // Boundary check and bounce-back logic
    if (newPosition.x != storePosition.x || newPosition.y != storePosition.y || newPosition.z != storePosition.z) {
        float random1 = hash(random_seeds[idx]);
        float random2 = hash(random_seeds[idx + 1]);

        random_seeds[idx] = random2;
        random_seeds[idx + 1] = random1;

        random1 = scaleToRange01(random1);
        random2 = scaleToRange01(random2);

        float theta = random1 * 2.0f * M_PI; // [0, 2π]
        // Uniformly sample cos(φ) from [-1, 1] and calculate φ from that
        float z = random2 * 2.0f - 1.0f; // z = cos(φ), [-1, 1]

        // Directly use z as the z-component, calculate the radius of the circle at z
        float r = sqrt(1.0f - z * z);

        // Calculate the x and y components based on r and theta
        float x = r * cos(theta);
        float y = r * sin(theta);

        float3 randomDirection = (float3)(x,y,z);
        newDirection = randomDirection;
    }

    // If not outside bounds update the spore's position
    spores[idx].position = newPosition;
    spores[idx].direction = newDirection;
}


__kernel void decay_trails(__global float* volume, __global const Settings* settings, const float delta_time) {
    // Calculate the 2D index of the current work item
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);
    uint z = (uint)get_global_id(2);

    // Check if the index is within the bounds of the image
    if (x < settings->simulation_size && y < settings->simulation_size && z < settings->simulation_size) {
        // Calculate the linear index of the pixel
        uint idx = z * settings->simulation_size * settings->simulation_size + y * settings->simulation_size + x;

        volume[idx] = max(0.0f, volume[idx] - settings->decay_speed * delta_time);
    }
}

