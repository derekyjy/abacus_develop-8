#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct AtomData
{
    double x, y, z;
    int type;
    int id;
};

struct MaterialCase
{
    std::string name;
    int nx, ny, nz;
    double lattice_x, lattice_y, lattice_z;
    double cutoff;
    int ntype;
    std::vector<std::tuple<int, double, double, double>> basis;
    int expected_atoms;
    std::string category;
};

std::vector<AtomData> generate_crystal_atoms(const MaterialCase& material)
{
    std::vector<AtomData> atoms;
    atoms.reserve(material.expected_atoms);
    int atom_id = 0;

    for (int ix = 0; ix < material.nx; ++ix)
        for (int iy = 0; iy < material.ny; ++iy)
            for (int iz = 0; iz < material.nz; ++iz)
                for (const auto& b : material.basis)
                {
                    if (atom_id >= material.expected_atoms)
                        break;
                    AtomData a;
                    a.x = (ix + std::get<1>(b)) * material.lattice_x;
                    a.y = (iy + std::get<2>(b)) * material.lattice_y;
                    a.z = (iz + std::get<3>(b)) * material.lattice_z;
                    a.type = std::get<0>(b);
                    a.id = atom_id++;
                    atoms.push_back(a);
                }
    return atoms;
}

double run_grid_serial_full(const std::vector<AtomData>& atoms, double cutoff, int& total_neighbors)
{
    double t0 = omp_get_wtime();

    int natoms = static_cast<int>(atoms.size());
    double cutoff2 = cutoff * cutoff;
    double grid_size = cutoff + 0.1;

    double x_min = atoms[0].x, x_max = atoms[0].x;
    double y_min = atoms[0].y, y_max = atoms[0].y;
    double z_min = atoms[0].z, z_max = atoms[0].z;
    for (const auto& a : atoms)
    {
        if (a.x < x_min)
            x_min = a.x;
        if (a.x > x_max)
            x_max = a.x;
        if (a.y < y_min)
            y_min = a.y;
        if (a.y > y_max)
            y_max = a.y;
        if (a.z < z_min)
            z_min = a.z;
        if (a.z > z_max)
            z_max = a.z;
    }

    int nx = static_cast<int>(std::ceil((x_max - x_min) / grid_size)) + 1;
    int ny = static_cast<int>(std::ceil((y_max - y_min) / grid_size)) + 1;
    int nz = static_cast<int>(std::ceil((z_max - z_min) / grid_size)) + 1;

    std::vector<std::vector<std::vector<std::vector<int>>>> grid(
        nx, std::vector<std::vector<std::vector<int>>>(ny, std::vector<std::vector<int>>(nz)));

    for (int i = 0; i < natoms; i++)
    {
        int bx = std::max(0, std::min(nx - 1, static_cast<int>(std::floor((atoms[i].x - x_min) / grid_size))));
        int by = std::max(0, std::min(ny - 1, static_cast<int>(std::floor((atoms[i].y - y_min) / grid_size))));
        int bz = std::max(0, std::min(nz - 1, static_cast<int>(std::floor((atoms[i].z - z_min) / grid_size))));
        grid[bx][by][bz].push_back(i);
    }

    total_neighbors = 0;

    for (int i = 0; i < natoms; i++)
    {
        int bxi = std::max(0, std::min(nx - 1, static_cast<int>(std::floor((atoms[i].x - x_min) / grid_size))));
        int byi = std::max(0, std::min(ny - 1, static_cast<int>(std::floor((atoms[i].y - y_min) / grid_size))));
        int bzi = std::max(0, std::min(nz - 1, static_cast<int>(std::floor((atoms[i].z - z_min) / grid_size))));

        for (int bx = 0; bx < nx; bx++)
        {
            for (int by = 0; by < ny; by++)
            {
                for (int bz = 0; bz < nz; bz++)
                {
                    for (int j : grid[bx][by][bz])
                    {
                        if (j <= i)
                            continue;
                        double dx2 = atoms[i].x - atoms[j].x;
                        double dy2 = atoms[i].y - atoms[j].y;
                        double dz2 = atoms[i].z - atoms[j].z;
                        if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < cutoff2)
                            total_neighbors++;
                    }
                }
            }
        }
    }

    return (omp_get_wtime() - t0) * 1000.0;
}

double run_grid_parallel_full(const std::vector<AtomData>& atoms,
                              double cutoff,
                              MPI_Comm comm,
                              int& total_neighbors)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double t0 = MPI_Wtime();

    int natoms = static_cast<int>(atoms.size());
    double cutoff2 = cutoff * cutoff;
    double grid_size = cutoff + 0.1;

    double x_min, x_max, y_min, y_max, z_min, z_max;
    if (rank == 0)
    {
        x_min = x_max = atoms[0].x;
        y_min = y_max = atoms[0].y;
        z_min = z_max = atoms[0].z;
        for (const auto& a : atoms)
        {
            if (a.x < x_min)
                x_min = a.x;
            if (a.x > x_max)
                x_max = a.x;
            if (a.y < y_min)
                y_min = a.y;
            if (a.y > y_max)
                y_max = a.y;
            if (a.z < z_min)
                z_min = a.z;
            if (a.z > z_max)
                z_max = a.z;
        }
    }

    MPI_Bcast(&natoms, 1, MPI_INT, 0, comm);
    MPI_Bcast(&x_min, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&x_max, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&y_min, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&y_max, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&z_min, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&z_max, 1, MPI_DOUBLE, 0, comm);

    int nx = static_cast<int>(std::ceil((x_max - x_min) / grid_size)) + 1;
    int ny = static_cast<int>(std::ceil((y_max - y_min) / grid_size)) + 1;
    int nz = static_cast<int>(std::ceil((z_max - z_min) / grid_size)) + 1;

    std::vector<double> atom_coords(natoms * 3);
    if (rank == 0)
    {
        for (int i = 0; i < natoms; i++)
        {
            atom_coords[i * 3] = atoms[i].x;
            atom_coords[i * 3 + 1] = atoms[i].y;
            atom_coords[i * 3 + 2] = atoms[i].z;
        }
    }
    MPI_Bcast(atom_coords.data(), natoms * 3, MPI_DOUBLE, 0, comm);

    std::vector<AtomData> local_atoms(natoms);
    for (int i = 0; i < natoms; i++)
    {
        local_atoms[i].x = atom_coords[i * 3];
        local_atoms[i].y = atom_coords[i * 3 + 1];
        local_atoms[i].z = atom_coords[i * 3 + 2];
        local_atoms[i].id = i;
    }

    std::vector<std::vector<std::vector<std::vector<int>>>> grid(
        nx, std::vector<std::vector<std::vector<int>>>(ny, std::vector<std::vector<int>>(nz)));

    for (int i = 0; i < natoms; i++)
    {
        int bx = std::max(0, std::min(nx - 1, static_cast<int>(std::floor((local_atoms[i].x - x_min) / grid_size))));
        int by = std::max(0, std::min(ny - 1, static_cast<int>(std::floor((local_atoms[i].y - y_min) / grid_size))));
        int bz = std::max(0, std::min(nz - 1, static_cast<int>(std::floor((local_atoms[i].z - z_min) / grid_size))));
        grid[bx][by][bz].push_back(i);
    }

    int base = natoms / size;
    int rem = natoms % size;
    int my_start = rank * base + std::min(rank, rem);
    int my_end = my_start + base + (rank < rem ? 1 : 0);

    int my_nb = 0;

#pragma omp parallel
    {
        int local_nb = 0;

#pragma omp for schedule(static) nowait
        for (int i = my_start; i < my_end; i++)
        {
            for (int bx = 0; bx < nx; bx++)
            {
                for (int by = 0; by < ny; by++)
                {
                    for (int bz = 0; bz < nz; bz++)
                    {
                        for (int j : grid[bx][by][bz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = local_atoms[i].x - local_atoms[j].x;
                            double dy2 = local_atoms[i].y - local_atoms[j].y;
                            double dz2 = local_atoms[i].z - local_atoms[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < cutoff2)
                                local_nb++;
                        }
                    }
                }
            }
        }

#pragma omp atomic
        my_nb += local_nb;
    }

    int global_nb = 0;
    MPI_Reduce(&my_nb, &global_nb, 1, MPI_INT, MPI_SUM, 0, comm);

    if (rank == 0)
        total_neighbors = global_nb;

    return (MPI_Wtime() - t0) * 1000.0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }

    const std::vector<MaterialCase> materials = {
        {"Al fcc", 10, 5, 5, 2.0, 2.0, 2.0, 1.5, 1,
         {{0, 0.0, 0.0, 0.0}, {0, 0.0, 0.5, 0.5}, {0, 0.5, 0.0, 0.5}, {0, 0.5, 0.5, 0.0}}, 1000,
         "PW / metal"},
        {"Si diamond", 10, 5, 5, 2.4, 2.4, 2.4, 1.25, 1,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.0, 0.5, 0.5},
          {0, 0.5, 0.0, 0.5},
          {0, 0.5, 0.5, 0.0},
          {0, 0.25, 0.25, 0.25},
          {0, 0.25, 0.75, 0.75},
          {0, 0.75, 0.25, 0.75},
          {0, 0.75, 0.75, 0.25}},
         2000, "LCAO / semiconductor"},
        {"NaCl", 15, 5, 5, 2.4, 2.4, 2.4, 1.35, 2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.0, 0.5, 0.5},
          {0, 0.5, 0.0, 0.5},
          {0, 0.5, 0.5, 0.0},
          {1, 0.5, 0.0, 0.0},
          {1, 0.0, 0.5, 0.0},
          {1, 0.0, 0.0, 0.5},
          {1, 0.5, 0.5, 0.5}},
         3000, "PW / ionic crystal"},
        {"TiO2 rutile", 10, 10, 7, 2.4, 2.4, 1.56, 1.25, 2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.5, 0.5, 0.5},
          {1, 0.305, 0.305, 0.0},
          {1, 0.695, 0.695, 0.0},
          {1, 0.805, 0.195, 0.5},
          {1, 0.195, 0.805, 0.5}},
         4200, "LCAO / complex oxide"},
    };

    if (rank == 0)
    {
        std::cout << "\n======================================================================" << std::endl;
        std::cout << "  ABACUS Material Neighbor Search - MPI+OpenMP Parallel Benchmark" << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << "  MPI Procs: " << size << "  |  OMP Threads/proc: " << num_threads << "  |  Total units: "
                  << size * num_threads << std::endl;
        std::cout << "  Algorithm:  Grid-based neighbor search (spatial grid)" << std::endl;
        std::cout << "======================================================================" << std::endl;
    }

    std::vector<std::tuple<std::string, int, double, double, double, double, std::string>> results;

    for (size_t midx = 0; midx < materials.size(); ++midx)
    {
        const auto& mat = materials[midx];

        std::vector<AtomData> atoms;
        if (rank == 0)
            atoms = generate_crystal_atoms(mat);

        double serial_ms = 0.0;
        int serial_nb = 0;
        if (rank == 0)
            serial_ms = run_grid_serial_full(atoms, mat.cutoff, serial_nb);

        MPI_Barrier(MPI_COMM_WORLD);

        double parallel_ms = 0.0;
        int parallel_nb = 0;
        parallel_ms = run_grid_parallel_full(atoms, mat.cutoff, MPI_COMM_WORLD, parallel_nb);

        if (rank == 0)
        {
            double avg_nb = static_cast<double>(serial_nb) / mat.expected_atoms;
            double sp = (parallel_ms > 0.0) ? serial_ms / parallel_ms : 0.0;
            results.push_back({mat.name, mat.expected_atoms, avg_nb, serial_ms, parallel_ms, sp, mat.category});
        }
    }

    if (rank == 0)
    {
        std::cout << "\n  "
                  << std::left << std::setw(16) << "Material"
                  << std::right << std::setw(8) << "Atoms"
                  << std::setw(8) << "AvgNB"
                  << std::setw(12) << "Serial(ms)"
                  << std::setw(12) << "Par(ms)"
                  << std::setw(8) << "Speedup"
                  << std::setw(9) << "Eff%"
                  << std::setw(22) << "Category" << std::endl;
        std::cout << "  " << std::string(95, '-') << std::endl;

        double total_s = 0, total_p = 0;
        for (const auto& r : results)
        {
            total_s += std::get<3>(r);
            total_p += std::get<4>(r);
            double sp = std::get<5>(r);
            std::cout << "  " << std::left << std::setw(16) << std::get<0>(r)
                      << std::right << std::setw(8) << std::get<1>(r)
                      << std::setw(8) << std::fixed << std::setprecision(1) << std::get<2>(r)
                      << std::setw(12) << std::fixed << std::setprecision(2) << std::get<3>(r)
                      << std::setw(12) << std::fixed << std::setprecision(2) << std::get<4>(r)
                      << std::setw(7) << std::fixed << std::setprecision(2) << sp << "x"
                      << std::setw(8) << std::fixed << std::setprecision(1) << (sp / (size * num_threads) * 100.0)
                      << "%" << std::setw(22) << std::get<6>(r) << std::endl;
        }
        std::cout << "  " << std::string(95, '-') << std::endl;
        double overall = total_s / total_p;
        std::cout << "  " << std::left << std::setw(16) << "OVERALL"
                  << std::right << std::setw(8) << "" << std::setw(8) << ""
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_s
                  << std::setw(12) << std::fixed << std::setprecision(2) << total_p
                  << std::setw(7) << std::fixed << std::setprecision(2) << overall << "x"
                  << std::setw(8) << std::fixed << std::setprecision(1) << (overall / (size * num_threads) * 100.0)
                  << "%" << std::endl;

        std::cout << "\n======================================================================" << std::endl;
        std::cout << "  Overall: " << std::fixed << std::setprecision(2) << overall << "x speedup, "
                  << std::fixed << std::setprecision(1) << (overall / (size * num_threads) * 100.0)
                  << "% efficiency" << std::endl;
        std::cout << "======================================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
