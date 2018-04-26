from __future__ import print_function
import numpy as np
from libc.math cimport sqrt
from Timer import Timer
from cython cimport boundscheck, wraparound
cimport cython

@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef direct_search(double[:, ::1] r, long num_parts, double h, double dp, int sim_dim, double x_min, double x_max,
                    double y_min, double y_max, double z_min, double z_max, int pbc, int pbc_x, int pbc_y, int pbc_z,
                    int[:, ::1] ipn_pairs, int trial_size):

    t = Timer()
    t.tic()

    cdef:
        int undersized = 0
        long part_i, part_j
        long ipn = 0
        double kr = 2 * h    # kernel radius.
        double tol = 0.001 * dp
        double r_ijx, r_ijy, r_ijz, rij, x_dist, y_dist, z_dist, periodic_xdist, periodic_ydist, periodic_zdist

    for part_i in range(1, num_parts + 1):
        for part_j in range(part_i, num_parts + 1):
            r_ijx = r[part_i, 0] - r[part_j, 0]
            r_ijy = r[part_i, 1] - r[part_j, 1]
            r_ijz = r[part_i, 2] - r[part_j, 2]

# ============= This next section of code calculates the absolute distance for periodic boundary conditions  ===========

            # Attention to the fact that the domain has to be at least four times as large the smooth length to avoid
            #  ambiguity in the interactions between particles.
            if pbc == 1:

                # =========================================== X-Coordinate =============================================
                if pbc_x == 1:

                    # Regular distance in the x-direction.
                    x_dist = abs(r_ijx)

                    # Check if particle j is not natural neighbor (only through periodicity).
                    if x_dist > kr:

                        # Find the periodic distance in the x-direction.
                        periodic_xdist = x_dist - x_max + x_min - dp
                        abs_p_xdist = abs(periodic_xdist)   # Absolute periodic distance in the x-direction.

                        # If periodic distance is larger than the regular distance, then keep regular distance in x-direction
                        if abs_p_xdist >= x_dist:
                            periodic_xdist = x_dist

                        # Based only on the new x-coordinates, check if part_j is a neighbor of the particles.
                        if abs_p_xdist <= kr + tol:

                            # The component is corrected with the periodic distance within the particle support.
                            r_ijx = periodic_xdist

                # ========================================== Y-Coordinate ==============================================
                if pbc_y == 1:
                    y_dist = abs(r_ijy)

                    if y_dist > kr:
                        periodic_ydist = y_dist - y_max + y_min - dp
                        abs_p_ydist = abs(periodic_ydist)

                        if abs_p_ydist >= y_dist:
                            periodic_ydist = y_dist

                        if abs_p_ydist <= kr + tol:
                            r_ijy = periodic_ydist

                # =========================================== Z-Coordinate =============================================

                if pbc_z == 1:
                    z_dist = abs(r_ijz)

                    if z_dist > kr:
                        periodic_zdist = z_dist - z_max + z_min - dp
                        abs_p_zdist = abs(periodic_zdist)

                        if abs_p_zdist >= z_dist:
                            periodic_zdist = z_dist

                        if abs_p_zdist <= kr + tol:
                            r_ijz = periodic_zdist

    # ==================================================================================================================

            # Distance between particles.
            rij = sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz)

            # Find particles that are neighbors and add information to the corresponding arrays.
            if rij <= kr + tol:

                # Number of neighbor pairs and neighbor pairs arrays
                ipn += 1

                # Test for size of the trial ipn_pairs array.
                if ipn == trial_size:
                    undersized = 1
                    break

                ipn_pairs[ipn, 0] = part_i
                ipn_pairs[ipn, 1] = part_j

        if undersized == 1:
            break

    t.toc('Direct search Cython')
    return ipn, undersized

# ======================================================================================================================
# =============================================== LINKED-LIST ==========================================================
# ======================================================================================================================
# This function receive an array of zeros with the number of cells and max num of neighbors as dimensions, and returns
#  it filled with each cell's neighbor indices. It assumes that there are at least 3 cells in the periodic direction!
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
cpdef create_grid(int num_cols, int num_rows, int num_layers, int[:, ::1] grid, int pbc, int pbc_x, int pbc_y,
                  int pbc_z, double x_max, double y_max, double z_max, double dp, double h):

    cdef:
        int row, col, lay, row2, col2, lay2, min_col, max_col, min_row, max_row, min_layer, max_layer, neigh_cell_num
        int pos, cell_num, row_idx, col_idx, lay_idx
        int rows[4]
        int cols[4]
        int layers[4]
        double kr = 2 * h
        double tol = 0.501 *  dp

    # This loop runs through all cells' indices.
    for lay in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):

                cell_num = col + row * num_cols + lay * num_cols * num_rows + 1

                # Read indices of neighbor cells, such that it can only read forward cells (which means, does not
                #  consider a cell with lower index).
                min_col = col - 1
                max_col = col + 1
                min_row = row - 1
                max_row = row + 1
                min_layer = lay - 1
                max_layer = lay + 1

                # Adjust the indices within the correct range for corner or edge cells.
                if min_col < 0:
                    min_col = 0
                if max_col + 1 > num_cols:
                    max_col = col

                if min_row < 0:
                    min_row = 0
                if max_row + 1 > num_rows:
                    max_row = row

                if min_layer < 0:
                    min_layer = 0
                if max_layer + 1 > num_layers:
                    max_layer = lay

                if pbc == 0:

                    # This loop is to find for each cell, its neighbors.
                    pos = 0
                    for lay2 in range(min_layer, max_layer + 1):
                        for row2 in range(min_row, max_row + 1):
                            for col2 in range(min_col, max_col + 1):

                                neigh_cell_num = col2 + row2 * num_cols + lay2 * num_cols * num_rows + 1
                                grid[cell_num, pos] = neigh_cell_num
                                pos +=1

                # Cell neighbor interval and list considering periodicity in the boundary.
                else:

                    # Initialize rows, cols, and layers arrays.
                    cols[0] = cols[1] = cols[2] = cols[3] = 0
                    rows[0] = rows[1] = rows[2] = rows[3] = 0
                    layers[0] = layers[1] = layers[2] = layers[3] = 0

                    # These indices give the estimated initial range of update and indexing of the rows, columns and
                    #  layers in the PBC loop.
                    if col == 0 or col == num_cols - 1:
                        col_idx = 2
                        cols[0] = min_col
                        cols[1] = max_col
                    else:
                        col_idx = 3
                        cols[0] = col - 1
                        cols[1] = col
                        cols[2] = col + 1

                    if row == 0 or row == num_rows - 1:
                        row_idx = 2
                        rows[0] = min_row
                        rows[1] = max_row
                    else:
                        row_idx = 3
                        rows[0] = row - 1
                        rows[1] = row
                        rows[2] = row + 1

                    if lay == 0 or lay == num_layers - 1:
                        lay_idx = 2
                        layers[0] = min_layer
                        layers[1] = max_layer
                    else:
                        lay_idx = 3
                        layers[0] = lay - 1
                        layers[1] = lay
                        layers[2] = lay + 1

                    # Adjust range for the case when one of the dimensions is unity, or in case one or more directions
                    #  are not periodic.
                    if num_cols == 1:
                        col_idx = 1
                        cols[0] = 0

                    if num_rows == 1:
                        row_idx = 1
                        rows[0] = 0

                    if num_layers == 1:
                        lay_idx = 1
                        layers[0] = 0

                    # Initialize the arrays for the corner cases when the grid extend past the periodic domain. Those
                    #  cases only apply to the very first, second to last, or last rows, columns, or layers.
                    if pbc_x == 1:
                        col_idx = 3

                        if num_cols * kr > x_max + tol:
                            if col == 0:
                                col_idx = 4
                                cols[2] = num_cols - 2
                                cols[3] = num_cols - 1
                            elif col == num_cols - 2:
                                col_idx = 4
                                cols[0] = 0
                                cols[1] = col - 1
                                cols[2] = col
                                cols[3] = col + 1

                        else:
                            if col == 0:
                                cols[0] = 0
                                cols[1] = 1
                                cols[2] = num_cols - 1
                            elif col == num_cols - 2:
                                cols[0] = col - 1
                                cols[1] = col
                                cols[2] = col + 1

                        if col == num_cols - 1:
                            cols[0] = 0
                            cols[1] = col - 1
                            cols[2] = col

                    if pbc_y == 1:
                        row_idx = 3

                        if num_rows * kr > y_max + tol:
                            if row == 0:
                                row_idx = 4
                                rows[2] = num_rows - 2
                                rows[3] = num_rows - 1
                            elif row == num_rows - 2:
                                row_idx = 4
                                rows[0] = 0
                                rows[1] = row - 1
                                rows[2] = row
                                rows[3] = row + 1

                        else:
                            if row == 0:
                                rows[0] = 0
                                rows[1] = 1
                                rows[2] = num_rows - 1
                            elif row == num_rows - 2:
                                rows[0] = row - 1
                                rows[1] = row
                                rows[2] = row + 1

                        if row == num_rows - 1:
                            rows[0] = 0
                            rows[1] = row - 1
                            rows[2] = row

                    if pbc_z == 1:
                        lay_idx = 3

                        if num_layers * kr > z_max + tol:
                            if lay == 0:
                                lay_idx = 4
                                layers[2] = num_layers - 2
                                layers[3] = num_layers - 1
                            elif lay == num_layers - 2:
                                lay_idx = 4
                                layers[0] = 0
                                layers[1] = lay - 1
                                layers[2] = lay
                                layers[3] = lay + 1

                        else:
                            if lay == 0:
                                layers[0] = 0
                                layers[1] = 1
                                layers[2] = num_layers - 1
                            elif lay == num_layers - 2:
                                layers[0] = lay - 1
                                layers[1] = lay
                                layers[2] = lay + 1

                        if lay == num_layers - 1:
                            layers[0] = 0
                            layers[1] = lay - 1
                            layers[2] = lay

                    # This loop is to find for each cell, its neighbors.
                    pos = 0
                    for lay2 in range(lay_idx):
                        for row2 in range(row_idx):
                            for col2 in range(col_idx):

                                neigh_cell_num = cols[col2] + rows[row2] * num_cols + layers[lay2] * num_cols * \
                                                 num_rows + 1
                                grid[cell_num, pos] = neigh_cell_num
                                pos +=1

# ======================================================================================================================
# This function allocates all particles to their respective cells in the grid. It returns two arrays, one with an array
#  of arrays of particles within each cell (cells_parts), and another array of particles' cell number (parts_cells).
@boundscheck(False)
@wraparound(False)
@cython.cdivision(True)
cpdef int allocate_particles(double[:, ::1] r, int num_cols, int num_rows, int num_layers, int num_cells, double h,
                             int num_parts, int[:, ::1] cells_parts, int[::1] parts_cells, double dp, int pbc):

    cdef:
        int row, col, lay, part_cell_num, part, cell_num, pos, max_index = 0
        double x, y, z, kr = 2 * h
        double tol = 0.001 * dp

    for part in range(1, num_parts + 1):
        # Particles coordinates.
        x = r[part, 0]
        y = r[part, 1]
        z = r[part, 2]

        # Find the column, row and layer for each particle.
        col = <int>((x - tol) / kr)
        row = <int>((y - tol) / kr)
        lay = <int>((z - tol) / kr)

        # Correct the column, row, or layer number for the case where the periodic boundary allows particles to move
        #  beyond the grid domain.
        if pbc == 1:
            col_max_idx = num_cols - 1
            row_max_idx = num_rows - 1
            lay_max_idx = num_layers - 1
            if col > col_max_idx: col = col_max_idx
            if row > row_max_idx: row = row_max_idx
            if lay > lay_max_idx: lay = lay_max_idx

        # Find the corresponding cell number for the particle and assign it to parts_cells.
        part_cell_num = col + row * num_cols + lay * num_cols * num_rows + 1
        parts_cells[part] = part_cell_num

        # Assign the current particle to its corresponding cell. The index pos keeps track of the next position in the
        #  array where to add the particle index.
        pos = cells_parts[part_cell_num, 0] + 1
        cells_parts[part_cell_num, pos] = part
        cells_parts[part_cell_num, 0] = pos

        if pos > max_index:
            max_index = pos

    return max_index + 1

# ======================================================================================================================
@boundscheck(False)
@wraparound(False)
cpdef linked_list(double[:, ::1] r, int[::1] parts_cells, int[:, ::1] cells_parts, int[:, ::1] grid, double h,
                  double dp, int sim_dim, long num_parts, int[:, ::1] ipn_pairs, int trial_size, double x_min,
                  double x_max, double y_min, double y_max, double z_min, double z_max, int pbc, int pbc_x, int pbc_y,
                  int pbc_z):

    t = Timer()
    t.tic()
    cdef:
        int undersized = 0
        long part_i, part_j, neighbor_cell, cell_num, i, j, ipn_init
        long ipn = 0
        double kr = 2 * h
        double tol = 0.001 * dp
        double r_ijx, r_ijy, r_ijz, rij, x_dist, y_dist, z_dist, periodic_xdist, periodic_ydist, periodic_zdist
        int[::1] cell_neighbors, neighbor_parts, temp

    for part_i in range(1, num_parts + 1):

        # First ipn index which gives ipn_pair[ , 0] = part_i.
        ipn_init = ipn + 1

        # Find the cell in which the current particle is.
        cell_num = parts_cells[part_i]

        # Find the particle's cell's neighbors.
        cell_neighbors = grid[cell_num]

        for i in range(cell_neighbors.shape[0]):
            neighbor_cell = cell_neighbors[i]

            # If zero, means it reached the end of the array, which by construction can have more slots than data.
            if neighbor_cell == 0:
                break

            # All particles within neighbor_cell.
            neighbor_parts = cells_parts[neighbor_cell]

            for j in range(neighbor_parts.shape[0]):
                part_j = neighbor_parts[j]

                # If zero, means it reached the end of the array.
                if part_j == 0:
                    break

                # If part j is smaller than part i, then disregard it as a pair and move to the next particle.
                if part_j < part_i:
                    continue

                # Vector distance components and distance.
                r_ijx = r[part_i, 0] - r[part_j, 0]
                r_ijy = r[part_i, 1] - r[part_j, 1]
                r_ijz = r[part_i, 2] - r[part_j, 2]

                # This next section of code calculates the absolute distance for periodic boundary conditions.
                # Attention to the fact that the domain has to be at least four times as large the smooth length
                #  to avoid ambiguity in the interactions between particles.
                if pbc == 1:

                    # ======================================= X-Coordinate =============================================
                    if pbc_x == 1:

                        # Regular distance in the x-direction.
                        x_dist = abs(r_ijx)

                        # Check if particle j is not natural neighbor (only through periodicity).
                        if x_dist > kr:

                            # Find the periodic distance in the x-direction.
                            periodic_xdist = x_dist - x_max + x_min - dp
                            abs_p_xdist = abs(periodic_xdist)

                            # If periodic distance is larger than the regular distance, then keep regular distance in
                            #  x-direction.
                            if abs_p_xdist >= x_dist:
                                periodic_xdist = x_dist

                            # Based only on the new x-coordinates, check if part_j is a neighbor of the particles.
                            if abs_p_xdist <= kr + tol:

                                # The component is corrected with the periodic distance within the particle support.
                                r_ijx = periodic_xdist

                    # ====================================== Y-Coordinate ==============================================
                    if pbc_y == 1:
                        y_dist = abs(r_ijy)

                        if y_dist > kr:
                            periodic_ydist = y_dist - y_max + y_min - dp
                            abs_p_ydist = abs(periodic_ydist)

                            if abs_p_ydist >= y_dist:
                                periodic_ydist = y_dist

                            if abs_p_ydist <= kr + tol:
                                r_ijy = periodic_ydist

                    # ======================================= Z-Coordinate =============================================

                    if pbc_z == 1:
                        z_dist = abs(r_ijz)

                        if z_dist > kr:
                            periodic_zdist = z_dist - z_max + z_min - dp
                            abs_p_zdist = abs(periodic_zdist)

                            if abs_p_zdist >= z_dist:
                                periodic_zdist = z_dist

                            if abs_p_zdist <= kr + tol:
                                r_ijz = periodic_zdist

                # ======================================================================================================

                rij = sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz)

                if rij <= kr + tol:
                    ipn += 1

                    # Test for size of the trial ipn_pairs array.
                    if ipn == trial_size:
                        undersized = 1
                        break

                    ipn_pairs[ipn, 0] = part_i
                    ipn_pairs[ipn, 1] = part_j

            # Break out of neighbor cells loop.
            if undersized == 1:
                break

        # Break out of the main part_i loop.
        if undersized == 1:
            break

        # This is to sort the particles such that part_j are stored in ascending order.
        temp = np.sort(np.ascontiguousarray(ipn_pairs[ipn_init:ipn + 1, 1]), kind='quick')
        ipn_pairs[ipn_init:ipn + 1, 1] = temp[:]

    t.toc('Linked-list NNPS Cython')
    return ipn, undersized
# ======================================================================================================================
