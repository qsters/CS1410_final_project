#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


typedef struct {
    float3 position;  // Use float2 for position
    float angle;
    float inclination;
} Spore;

typedef struct {
    uint spore_count;
    uint screen_size;
    float spore_speed;
    float decay_speed;
    float turn_speed;
    float sensor_distance;
} Settings;

//// Function prototypes
uint hash(uint state);
float scaleToRange01(uint state);
//float sense(__global Spore* spore, __global uchar* image, __global const Settings* settings, float sensorAngleOffset);

//float sense(__global Spore* spore, __global uchar* image, __global const Settings* settings, float sensorAngleOffset) {
//    float sensorAngle = spore->angle + sensorAngleOffset;
//    float2 sensorDir = (float2)(cos(sensorAngle), sin(sensorAngle));
//
//    // Calculate the sampling position
//    float2 samplePos = spore->position + sensorDir * settings->sensor_distance;
//
//    // Clamp the sampling position to be within the image bounds
//    uint sampleX = clamp((uint)samplePos.x, (uint)0, settings->screen_width - (uint)1);
//    uint sampleY = clamp((uint)samplePos.y, (uint)0, settings->screen_height - (uint)1);
//
//    // Calculate the linear index in the image array
//    uint idx = (sampleY * settings->screen_width + sampleX) * 4; // 4 bytes per pixel
//
//    // Access the green component of the pixel,
//    float sensedAlpha = (float)image[idx + (uint)2] / 255.0f; // Normalize the alpha value to [0, 1]
//
//    return sensedAlpha;
//}

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
    if (x < settings->screen_size && y < settings->screen_size && z < settings->screen_size) {
        uint volume_idx = z * settings->screen_size * settings->screen_size + y * settings->screen_size + x;
        volume[volume_idx] = 1.0f; // Place a 1 at the position of the spore
    }
}

__kernel void move_spores(__global Spore* spores, __global float* volume, __global uint* random_seeds, __global const Settings* settings, const float delta_time) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    float azimuth = spores[idx].angle;
    float inclination = spores[idx].inclination;

    float speed = 1.0f;

    float3 direction = (float3)(sin(inclination) * cos(azimuth),
                                       sin(inclination) * sin(azimuth),
                                       cos(inclination)) * speed;

    float3 newPosition = spores[idx].position + direction;

    // Boundary check and bounce-back logic
    if (newPosition.x < 0 || newPosition.x >= settings->screen_size || newPosition.y < 0 || newPosition.y >= settings->screen_size || newPosition.z < 0 || newPosition.z >= settings->screen_size) {
        uint random = hash(random_seeds[idx]);
        uint random2 = hash(random_seeds[idx] + random);
        random_seeds[idx] = random2;

        float randomAngle = scaleToRange01(random) * 2.0f * M_PI; // M_PI is the PI constant in OpenCL
        float randomInclination = scaleToRange01(random2) * 2.0f * M_PI; // M_PI is the PI constant in OpenCL

        // Ensure newPosition is within bounds
        newPosition.x = clamp(newPosition.x, 0.0f, (float)(settings->screen_size - 1));
        newPosition.y = clamp(newPosition.y, 0.0f, (float)(settings->screen_size - 1));
        newPosition.z = clamp(newPosition.z, 0.0f, (float)(settings->screen_size - 1));


        spores[idx].angle = randomAngle;
        spores[idx].inclination = randomInclination;
    }

    // If not outside bounds update the spore's position
    spores[idx].position = newPosition;
}

__kernel void decay_trails(__global float* volume, __global const Settings* settings, const float delta_time) {
    // Calculate the 2D index of the current work item
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);
    uint z = (uint)get_global_id(2);

    // Check if the index is within the bounds of the image
    if (x < settings->screen_size && y < settings->screen_size && z < settings->screen_size) {
        // Calculate the linear index of the pixel
        uint idx = z * settings->screen_size * settings->screen_size + y * settings->screen_size + x;

        volume[idx] = max(0.0f, volume[idx] - settings->decay_speed * delta_time);
    }
}

