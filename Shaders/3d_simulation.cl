#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


typedef struct {
    float3 position;  // Use float2 for position
    float azimuth;
    float inclination;
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
uint hash(uint state);
float scaleToRange01(uint state);
float3 sph_to_car(float azimuth, float inclination);
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

float3 sph_to_car(float azimuth, float inclination) {
    return (float3)(sin(inclination) * cos(azimuth),
                     sin(inclination) * sin(azimuth),
                     cos(inclination));
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
    volume[idx] = 1.0; // Assign a value to indicate a sensor's position
}


uint hash(uint state) {
    state ^= 2747636419u;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    return state;
}

float scaleToRange01(uint state) {
    // Explicitly specify the divisor as a float to ensure floating-point division.
    return (float)state / 4294967295.0f;
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

    float3 upVector = (float3)(1.0f, 0.0f, 0.0f); // Global up
    float3 sporeDirection = sph_to_car(spores[idx].azimuth, spores[idx].inclination); // Assuming spores[].direction is already defined

    // Generate the local right vector
    float3 rightVector = cross(sporeDirection, upVector);
    if (length(rightVector) == 0) { // Handle parallel or antiparallel direction
        // Fallback or adjust rightVector
        rightVector = (float3)(1.0f, 0.0f, 0.0f);
    }
    rightVector = normalize(rightVector);

    // Generate local up vector based on right and forward vectors
    float3 upVectorAdjusted = cross(rightVector, sporeDirection);
    upVectorAdjusted = normalize(upVectorAdjusted);

    draw_sensor(&spores[idx], volume, settings, sporeDirection, sporeDirection);
//    draw_sensor(&spores[idx], volume, settings, rightVector, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, -rightVector, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, upVectorAdjusted, sporeDirection);
    draw_sensor(&spores[idx], volume, settings, -upVectorAdjusted, sporeDirection);
}



__kernel void move_spores(__global Spore* spores, __global float* volume, __global uint* random_seeds, __global const Settings* settings, const float delta_time) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    float3 upVector = (float3)(-1.0f, 0.0f, 0.0f); // Global up
    float3 sporeDirection = sph_to_car(spores[idx].azimuth, M_PI / 2); // Assuming spores[].direction is already defined

    // Generate the local right vector
    float3 rightVector = cross(sporeDirection, upVector);
    if (length(rightVector) == 0) { // Handle parallel or antiparallel direction
        // Fallback or adjust rightVector
        rightVector = (float3)(1.0f, 0.0f, 0.0f);
    }
    rightVector = normalize(rightVector);

    // Generate local up vector based on right and forward vectors
    float3 upVectorAdjusted = cross(rightVector, sporeDirection);
    upVectorAdjusted = normalize(upVectorAdjusted);

    float forwardWeight = sense(&spores[idx], volume, settings, sporeDirection, sporeDirection);
    float rightWeight = sense(&spores[idx], volume, settings, rightVector, sporeDirection);
    float leftWeight = sense(&spores[idx], volume, settings, -rightVector, sporeDirection);
    float upWeight = sense(&spores[idx], volume, settings, upVectorAdjusted, sporeDirection);
    float downWeight = sense(&spores[idx], volume, settings, -upVectorAdjusted, sporeDirection);

    // Initialize the change in azimuth and inclination
    float azimuthDelta = 0.1;
    float inclinationDelta = 0.0;

//    if (forwardWeight < rightWeight || forwardWeight < leftWeight) {
//        // Decide turning based on left and right weights
//        if (rightWeight > leftWeight) {
//            azimuthDelta -= 1; // Turn right
//        } else if (leftWeight > rightWeight) {
//            azimuthDelta += 1; // Turn left
//        }
//    }
//
//    if (forwardWeight < upWeight || forwardWeight < downWeight) {
//        // Decide inclination based on up and down weights
//        if (upWeight > downWeight) {
//            inclinationDelta -= 1; // Move upwards
//        } else if (downWeight > upWeight) {
//            inclinationDelta += 1; // Move downwards
//        }
//    }

    float newAzimuth = spores[idx].azimuth + azimuthDelta * settings->turn_speed * delta_time;
    float newInclination = spores[idx].inclination + inclinationDelta * settings->turn_speed * delta_time;

    float3 newDirection = (float3)(sin(newInclination) * cos(newAzimuth),
                                       sin(newInclination) * sin(newAzimuth),
                                       cos(newInclination));

    float3 newPosition = spores[idx].position + newDirection * settings->spore_speed * delta_time;

    // Boundary check and bounce-back logic
    if (newPosition.x < 0 || newPosition.x >= settings->simulation_size ||
        newPosition.y < 0 || newPosition.y >= settings->simulation_size ||
        newPosition.z < 0 || newPosition.z >= settings->simulation_size) {

        // Reverse the direction by adjusting azimuth and inclination
        // This is a conceptual reversal; in practice, you may need a more nuanced approach
        newAzimuth += M_PI; // Adding π to the azimuth effectively reverses the direction in the XY plane
        newInclination += M_PI / 2;
        // For inclination, consider what reversal means in your application context
        // This might not be necessary if you simply reverse the XY direction and maintain the current vertical direction

        // Ensure newAzimuth is within bounds [0, 2π]

        // Recalculate newDirection with updated azimuth (and possibly inclination)
        newDirection = (float3)(sin(newInclination) * cos(newAzimuth),
                                sin(newInclination) * sin(newAzimuth),
                                cos(newInclination));

        // Re-calculate newPosition to ensure it's within bounds
        newPosition = spores[idx].position + newDirection * settings->spore_speed * delta_time;
        newPosition.x = clamp(newPosition.x, 0.0f, (float)(settings->simulation_size - 1));
        newPosition.y = clamp(newPosition.y, 0.0f, (float)(settings->simulation_size - 1));
        newPosition.z = clamp(newPosition.z, 0.0f, (float)(settings->simulation_size - 1));
    }

    // If not outside bounds update the spore's position
    spores[idx].position = newPosition;
    spores[idx].azimuth = newAzimuth;
    spores[idx].inclination = M_PI / 2;
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

