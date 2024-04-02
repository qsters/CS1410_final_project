#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


typedef struct {
    float2 position;  // Use float2 for position
    float angle;
} Spore;

typedef struct {
    uint spore_count;
    uint screen_height;
    uint screen_width;
    float spore_speed;
    float decay_speed;
    float turn_speed;
    float sensor_distance;
} Settings;

// Function prototypes
uint hash(uint state);
float scaleToRange01(uint state);
float sense(__global Spore* spore, __global uchar* image, __global const Settings* settings, float sensorAngleOffset);

float sense(__global Spore* spore, __global uchar* image, __global const Settings* settings, float sensorAngleOffset) {
    float sensorAngle = spore->angle + sensorAngleOffset;
    float2 sensorDir = (float2)(cos(sensorAngle), sin(sensorAngle));

    // Calculate the sampling position
    float2 samplePos = spore->position + sensorDir * settings->sensor_distance;

    // Clamp the sampling position to be within the image bounds
    uint sampleX = clamp((uint)samplePos.x, (uint)0, settings->screen_width - (uint)1);
    uint sampleY = clamp((uint)samplePos.y, (uint)0, settings->screen_height - (uint)1);

    // Calculate the linear index in the image array
    uint idx = (sampleY * settings->screen_width + sampleX) * 4; // 4 bytes per pixel

    // Access the green component of the pixel,
    float sensedAlpha = (float)image[idx + (uint)2] / 255.0f; // Normalize the alpha value to [0, 1]

    return sensedAlpha;
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

__kernel void draw_spores(__global uchar* image, __global Spore* spores, __global const Settings* settings) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    Spore s = spores[idx];
    uint x = (int)(s.position.x);
    uint y = (int)(s.position.y);

    // Ensure the coordinates are within the image bounds
    if (x < settings->screen_width && y < settings->screen_height) {
        // Assuming the image is a 1D array in row-major order and each pixel is 4 bytes (RGBA)
        int image_idx = 4 * (y * settings->screen_width + x);
        image[image_idx] = (uchar)((float)x / (float)settings->screen_width * 255);  // R
        image[image_idx + 1] = (uchar)((float)y / (float)settings->screen_height * 255);  // G
        image[image_idx + 2] = 255; // B
        image[image_idx + 3] = 255; // A (fully opaque)
    }
}

// Include hash and scaleToRange01 functions here

__kernel void move_spores(__global Spore* spores, __global uchar* image,__global uint* random_seeds, __global const Settings* settings, const float delta_time) {
    uint idx = (uint)get_global_id(0);

    if (idx >= settings->spore_count) {
        return;
    }

    float angle = spores[idx].angle;

    float forwardWeight = sense(&spores[idx], image, settings, 0.0f);
    float rightWeight = sense(&spores[idx], image, settings, -M_PI / 4.0f);
    float leftWeight = sense(&spores[idx], image, settings, M_PI / 4.0f);

    // Starting the change with some random amount
    float turningDelta = 0;

    // Forward strongest, keep going forward
    if (forwardWeight > rightWeight && forwardWeight > leftWeight)
    {
        turningDelta += 0;
    }
    else if (rightWeight > leftWeight) // turn right
    {
        turningDelta -= 1;
    }
    else if (leftWeight > rightWeight) // turn left
    {
        turningDelta += 1;
    }

    float newAngle = angle + turningDelta * settings->turn_speed * delta_time * M_PI;

    float2 direction = (float2)(cos(angle), sin(angle)) * settings->spore_speed * delta_time;
    float2 newPosition = spores[idx].position + direction;

    // Boundary check and bounce-back logic
    if (newPosition.x < 0 || newPosition.x >= settings->screen_width || newPosition.y < 0 || newPosition.y >= settings->screen_height) {
        uint random = hash(random_seeds[idx]);
        random_seeds[idx] = random;

        float randomAngle = scaleToRange01(random) * 2.0f * M_PI; // M_PI is the PI constant in OpenCL

        // Ensure newPosition is within bounds
        newPosition.x = clamp(newPosition.x, 0.0f, (float)(settings->screen_width - 1));
        newPosition.y = clamp(newPosition.y, 0.0f, (float)(settings->screen_height - 1));

        // Update the spore's angle and position with the new values
        newAngle = randomAngle;
    }

    // If not outside bounds update the spore's position
    spores[idx].position = newPosition;
    spores[idx].angle = newAngle;
}

__kernel void fade_image(__global uchar* image, __global const Settings* settings, const uchar decay_amount) {
    // Calculate the 2D index of the current work item
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);

    // Calculate the linear index of the pixel
    uint idx = 4 * (y * settings->screen_width + x); // 4 bytes per pixel (RGBA)

    // Check if the index is within the bounds of the image
    if ((uint)x < settings->screen_width && (uint)y < settings->screen_height) {
        image[idx] = (uchar)max(0, image[idx] - decay_amount);  // R
        image[idx + 1] = (uchar)max(0, image[idx + 1] - decay_amount); // G
        image[idx + 2] = (uchar)max(0, image[idx + 2] - decay_amount); // B
        // Alpha channel remains unchanged image[idx + 3]
    }
}

