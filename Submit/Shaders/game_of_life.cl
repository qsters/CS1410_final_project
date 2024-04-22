__kernel void game_of_life(__global uchar* pixels, __global uchar* new_pixels, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int num_alive = 0;
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue; // Skip the cell itself
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int index = 4 * (ny * width + nx); // Calculate the index for the RGBA values of the pixel
                if (pixels[index] > 0) {
                    num_alive++;
                }
            }
        }
    }

    int index = 4 * (y * width + x); // Calculate the index for the RGBA values of the pixel
    uchar cell_r = pixels[index];
    uchar cell_g = pixels[index + 1];
    uchar cell_b = pixels[index + 2];
    uchar new_cell_r = 0;
    uchar new_cell_g = 0;
    uchar new_cell_b = 0;

    if ((cell_r > 0 || cell_g > 0 || cell_b > 0) && (num_alive == 2 || num_alive == 3)) {
        new_cell_r = cell_r; // Cell stays alive
        new_cell_g = cell_g;
        new_cell_b = cell_b;
    } else if ((cell_r == 0 && cell_g == 0 && cell_b == 0) && num_alive == 3) {
        new_cell_r = 255; // Cell becomes alive
        new_cell_g = 255;
        new_cell_b = 255;
    }

    new_pixels[index] = new_cell_r;
    new_pixels[index + 1] = new_cell_g;
    new_pixels[index + 2] = new_cell_b;
    // Alpha channel remains unchanged (not modified)
}
