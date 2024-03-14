__kernel void game_of_life(__global int* grid, __global int* new_grid, const unsigned int width, const unsigned int height) {
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
                num_alive += grid[ny * width + nx];
            }
        }
    }

    int index = y * width + x;
    int cell_is_alive = grid[index];
    int new_state = 0;

    if (cell_is_alive && (num_alive == 2 || num_alive == 3)) {
        new_state = 1; // Cell stays alive
    } else if (!cell_is_alive && num_alive == 3) {
        new_state = 1; // Cell becomes alive
    }

    new_grid[index] = new_state;
}
