__author__ = 'alomir'

import numpy as np
from Timer import Timer


class DirectSearch:

    def __init__(self):
        pass

    @staticmethod
    def neighbor_search(r, num_parts, h, dp, kernel_obj, sim_dim, x_min, x_max, y_min, y_max, z_min, z_max, pbc_x,
                        pbc_y, pbc_z):

        t = Timer()
        t.tic()

        parts = np.arange(1, num_parts + 1, dtype=int)

        # Creates an array with repeated indices for each particle in sequence: (1,1,...,1,2,2,...,2,...,N,N,...,N)
        parti = np.repeat(parts, num_parts)

        # Creates an array if all particles in order num_parts times: (1,2,3,4,...,N,1,2,3,4,...,N,1,...,N)
        partj = np.tile(parts, num_parts)

        r_ij = r[parti] - r[partj]

# ============= This next section of code calculates the absolute distance for periodic boundary conditions  ===========

        # Attention to the fact that the domain has to be at least four times as large the smooth length to avoid
        #  ambiguity in the interactions between particles.

        # =========================================== X-Coordinate =====================================================

        if pbc_x == 1:

            # Find the periodic distance in the x-direction.
            periodic_xdist = np.abs(r_ij[:, 0]) - (x_max - x_min + dp)

            # Regular distance in the x-direction.
            x_dist = np.abs(r_ij[:, 0])

            # If periodic distance is larger than the regular distance, then keep regular distance in x-direction
            more = np.where(np.abs(periodic_xdist) > x_dist)[0]
            periodic_xdist[more] = x_dist[more]

            # Based only on the new x-coordinates, find possible neighbors of the particles.
            neighbors2 = np.where(np.abs(periodic_xdist) < 2 * h)[0]

            # The vector is corrected for those with periodic distances within the particle support.
            # The sign function is used to correct the sign of the vector r_ij.
            r_ij[neighbors2, 0] = periodic_xdist[neighbors2] * np.sign(r_ij[neighbors2, 0])

        # ========================================== Y-Coordinate ======================================================

        if pbc_y == 1:
            periodic_ydist = np.abs(r_ij[:, 1]) - (y_max - y_min + dp)
            y_dist = np.abs(r_ij[:, 1])
            more = np.where(np.abs(periodic_ydist) > y_dist)[0]
            periodic_ydist[more] = y_dist[more]
            neighbors2 = np.where(np.abs(periodic_ydist) < 2 * h)[0]
            r_ij[neighbors2, 1] = periodic_ydist[neighbors2] * np.sign(r_ij[neighbors2, 1])

        # =========================================== Z-Coordinate =====================================================

        if pbc_z == 1:
            periodic_zdist = np.abs(r_ij[:, 2]) - (z_max - z_min + dp)
            z_dist = np.abs(r_ij[:, 2])
            more = np.where(np.abs(periodic_zdist) > z_dist)[0]
            periodic_zdist[more] = z_dist[more]
            neighbors2 = np.where(np.abs(periodic_zdist) < 2 * h)[0]
            r_ij[neighbors2, 2] = periodic_zdist[neighbors2] * np.sign(r_ij[neighbors2, 2])

# ======================================================================================================================

        rij = np.linalg.norm(r_ij, ord=None, axis=1)
        rij = np.reshape(rij, (np.size(rij), 1))

        # Find particles that are neighbors. This applies for non-periodic boundary conditions.
        neighbors_indices = np.where(rij[:] < 2 * h)[0]
        num_neighbors = np.size(neighbors_indices)

        # Get rid of all pairs that are not neighbors.
        parti = parti[neighbors_indices]
        partj = partj[neighbors_indices]
        r_ij = r_ij[neighbors_indices]
        rij = rij[neighbors_indices]

        ipn_pairs = np.zeros((num_neighbors, 2), dtype=int)
        ipn_pairs[:, 0] = parti[:]
        ipn_pairs[:, 1] = partj[:]

        # This part is just so the pairs and the corresponding distance vectors and norms start at index 1.
        zero = np.zeros((1, 2), dtype=int)
        zero2 = np.zeros((1, 3), dtype=np.float64)
        zero3 = np.zeros((1, 1), dtype=np.float64)

        ipn_pairs = np.concatenate((zero, ipn_pairs))
        ipn = np.size(ipn_pairs[:, 0]) - 1
        r_ij = np.concatenate((zero2, r_ij))
        rij = np.concatenate((zero3, rij))

        w = np.zeros((ipn + 1, 1))
        dw = np.zeros((ipn + 1, 3))
        kernel_obj.kernel_calcs(r_ij, rij, h, sim_dim, w, dw)

        t.toc('Direct search NNPS algorithm and kernel calculation')

        return ipn, ipn_pairs, w, dw, r_ij, rij

# ======================================================================================================================
# ========================================== LINKED-LIST ===============================================================
# ======================================================================================================================


class LinkedListSearch:

    def __init__(self):
        pass

    @staticmethod
    def create_grid(num_cols, num_rows, num_layers):

        # t = Timer()
        # t.tic()

        # This is called "grid" in the Solver.py module.
        cells_neighbors_list = [np.zeros(1, dtype=np.int)]

        # This loop runs through all cells' indices.
        for layer in range(num_layers):
            for row in range(num_rows):
                for col in range(num_cols):

                    # Read indices of neighbor cells, such that it can only read forward cells (which means, does not
                    #  consider a cell with lower index).
                    min_col = col - 1
                    max_col = col + 1
                    min_row = row - 1
                    max_row = row + 1
                    min_layer = layer - 1
                    max_layer = layer + 1

                    # This is to adjust next loop's interval.
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
                        max_layer = layer

                    # Temporary list of neighbor cells for the current cell.
                    temp = []

                    # This loop is to find for each cell, its neighbors.
                    for layer2 in range(min_layer, max_layer + 1):
                        for row2 in range(min_row, max_row + 1):
                            for col2 in range(min_col, max_col + 1):

                                neighbor_cell_num = col2 + row2 * num_cols + layer2 * num_cols * num_rows + 1
                                temp.append(neighbor_cell_num)

                    # Add the neighbors of the current cell to the list neighbors.
                    cells_neighbors_list.append(np.asarray(temp))

        # t.toc('Creating cell grid took ')

        return np.asarray(cells_neighbors_list)

# ======================================================================================================================
    @staticmethod
    def allocate_particles(r, num_cols, num_rows, num_cells, h):

        # t = Timer()
        # t.tic()

        # List of a list of particles for each cell.
        cell_part_list = [np.zeros(1, dtype=np.int)]

        # Particles coordinates.
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]

        cell_side = 2 * h

        # Find the column, row and layer for each particle.
        col = (x / cell_side).astype(int)
        row = (y / cell_side).astype(int)
        layer = (z / cell_side).astype(int)

        # Find the corresponding cell number for each particle.
        part_cell_list = col + row * num_cols + layer * num_cols * num_rows + 1
        part_cell_list[0] = -1.0

        for cell_num in range(1, num_cells + 1):

            # Particles within current cell.
            part_indices = np.where(part_cell_list[:] == cell_num)[0]

            # Append these particles to the list corresponding to the current cell.
            cell_part_list.append(part_indices)

        # t.toc('Allocating particles took ')

        return np.asarray(cell_part_list), part_cell_list

# ======================================================================================================================
    @staticmethod
    def neighbor_search(r, part_cell_list, cell_part_list, cell_neighbor_list, h, kernel_obj, sim_dim, num_parts):

        t = Timer()
        t.tic()

        # Array that carries the pairs of neighbors, (i,j).
        ipn_pairs = np.zeros((1, 2), dtype=np.int)

        for part in range(1, num_parts + 1):

                # Find the cell in which the current particle is.
                part_cell_num = part_cell_list[part]

                # Find the particle's cell neighbors.
                cell_neighbors = np.asarray(cell_neighbor_list[part_cell_num])

                # Create an array containing all particles inside all the neighbor cells.
                partj = np.concatenate((cell_part_list[cell_neighbors]))

                # Vector and distance between particles.
                r_ij = r[part] - r[partj]
                rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

                # Find particles within the kernel of the current particle and sort them.
                neighbors = np.where(rij < 2 * h)[0]
                partj = np.sort(partj[neighbors], kind='mergesort')
                size = np.size(neighbors)

                # Create the array of pairs (i,j) for the current particle
                ipn_pairs_i = np.zeros((size, 2), dtype=np.int)
                ipn_pairs_i[:, 0] = part
                ipn_pairs_i[:, 1] = partj

                # Append the array to the global array of neighbor pairs.
                ipn_pairs = np.concatenate((ipn_pairs, ipn_pairs_i))

        # Number of unique pairs.
        ipn = np.size(ipn_pairs[:, 0]) - 1

        # Update vector and norm of distance.
        r_ij = r[ipn_pairs[:, 0]] - r[ipn_pairs[:, 1]]
        rij = np.linalg.norm(r_ij, ord=None, axis=1, keepdims=True)

        # Calculate kernel and kernel derivative.
        w = np.zeros((ipn + 1, 1))
        dw = np.zeros((ipn + 1, 3))
        kernel_obj.kernel_calcs(r_ij, rij, h, sim_dim, w, dw)

        t.toc('Finding neighbors and kernel calculation')

        return ipn, ipn_pairs, w, dw, r_ij, rij

# ======================================================================================================================
