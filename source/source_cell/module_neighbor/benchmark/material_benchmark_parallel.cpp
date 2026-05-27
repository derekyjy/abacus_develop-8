#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
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
    double expected_average_neighbors;
    std::string category;
};

std::vector<AtomData> generate_crystal_atoms(const MaterialCase& material)
{
    std::vector<AtomData> atoms;
    atoms.reserve(material.expected_atoms);
    int atom_id = 0;

    for (int ix = 0; ix < material.nx; ++ix)
    {
        for (int iy = 0; iy < material.ny; ++iy)
        {
            for (int iz = 0; iz < material.nz; ++iz)
            {
                for (const auto& basis : material.basis)
                {
                    if (atom_id >= material.expected_atoms)
                        break;
                    const int t = std::get<0>(basis);
                    const double fx = std::get<1>(basis);
                    const double fy = std::get<2>(basis);
                    const double fz = std::get<3>(basis);
                    AtomData a;
                    a.x = (ix + fx) * material.lattice_x;
                    a.y = (iy + fy) * material.lattice_y;
                    a.z = (iz + fz) * material.lattice_z;
                    a.type = t;
                    a.id = atom_id++;
                    atoms.push_back(a);
                }
            }
        }
    }
    return atoms;
}

double run_grid_search_serial(const std::vector<AtomData>& atoms, double cutoff, int& total_neighbors)
{
    double start = omp_get_wtime();

    int natoms = static_cast<int>(atoms.size());
    double cutoff2 = cutoff * cutoff;
    double grid_size = cutoff + 0.1;

    double x_min = atoms[0].x, x_max = atoms[0].x;
    double y_min = atoms[0].y, y_max = atoms[0].y;
    double z_min = atoms[0].z, z_max = atoms[0].z;
    for (const auto& a : atoms)
    {
        x_min = std::min(x_min, a.x);
        x_max = std::max(x_max, a.x);
        y_min = std::min(y_min, a.y);
        y_max = std::max(y_max, a.y);
        z_min = std::min(z_min, a.z);
        z_max = std::max(z_max, a.z);
    }

    int nx = static_cast<int>(std::ceil((x_max - x_min) / grid_size)) + 1;
    int ny = static_cast<int>(std::ceil((y_max - y_min) / grid_size)) + 1;
    int nz = static_cast<int>(std::ceil((z_max - z_min) / grid_size)) + 1;

    std::vector<std::vector<std::vector<std::vector<int>>>> grid(
        nx, std::vector<std::vector<std::vector<int>>>(
            ny, std::vector<std::vector<int>>(nz)));

    for (int i = 0; i < natoms; i++)
    {
        int bx = static_cast<int>(std::floor((atoms[i].x - x_min) / grid_size));
        int by = static_cast<int>(std::floor((atoms[i].y - y_min) / grid_size));
        int bz = static_cast<int>(std::floor((atoms[i].z - z_min) / grid_size));
        bx = std::max(0, std::min(bx, nx - 1));
        by = std::max(0, std::min(by, ny - 1));
        bz = std::max(0, std::min(bz, nz - 1));
        grid[bx][by][bz].push_back(i);
    }

    int search_range = static_cast<int>(std::ceil(cutoff / grid_size));
    std::vector<std::vector<int>> neighbors(natoms);
    total_neighbors = 0;

    for (int i = 0; i < natoms; i++)
    {
        int bxi = static_cast<int>(std::floor((atoms[i].x - x_min) / grid_size));
        int byi = static_cast<int>(std::floor((atoms[i].y - y_min) / grid_size));
        int bzi = static_cast<int>(std::floor((atoms[i].z - z_min) / grid_size));
        bxi = std::max(0, std::min(bxi, nx - 1));
        byi = std::max(0, std::min(byi, ny - 1));
        bzi = std::max(0, std::min(bzi, nz - 1));

        for (int dx = -search_range; dx <= search_range; dx++)
        {
            for (int dy = -search_range; dy <= search_range; dy++)
            {
                for (int dz = -search_range; dz <= search_range; dz++)
                {
                    int bx = bxi + dx;
                    int by = byi + dy;
                    int bz = bzi + dz;
                    if (bx < 0 || bx >= nx || by < 0 || by >= ny || bz < 0 || bz >= nz)
                        continue;

                    for (int j : grid[bx][by][bz])
                    {
                        if (j <= i)
                            continue;
                        double dx2 = atoms[i].x - atoms[j].x;
                        double dy2 = atoms[i].y - atoms[j].y;
                        double dz2 = atoms[i].z - atoms[j].z;
                        double dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                        if (dist2 < cutoff2)
                        {
                            total_neighbors++;
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    return (end - start) * 1000.0;
}

double run_grid_search_mpi_omp(const std::vector<AtomData>& atoms,
                               double cutoff,
                               MPI_Comm comm,
                               int& total_neighbors)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double mpi_start = MPI_Wtime();

    int natoms = static_cast<int>(atoms.size());
    MPI_Bcast(&natoms, 1, MPI_INT, 0, comm);

    std::vector<double> atom_data;
    if (rank == 0)
    {
        atom_data.resize(natoms * 3);
        for (int i = 0; i < natoms; i++)
        {
            atom_data[i * 3] = atoms[i].x;
            atom_data[i * 3 + 1] = atoms[i].y;
            atom_data[i * 3 + 2] = atoms[i].z;
        }
    }
    else
    {
        atom_data.resize(natoms * 3);
    }
    MPI_Bcast(atom_data.data(), natoms * 3, MPI_DOUBLE, 0, comm);

    std::vector<AtomData> local_atoms(natoms);
    for (int i = 0; i < natoms; i++)
    {
        local_atoms[i].x = atom_data[i * 3];
        local_atoms[i].y = atom_data[i * 3 + 1];
        local_atoms[i].z = atom_data[i * 3 + 2];
        local_atoms[i].id = i;
    }

    double cutoff2 = cutoff * cutoff;
    double grid_size = cutoff + 0.1;

    double x_min = local_atoms[0].x, x_max = local_atoms[0].x;
    double y_min = local_atoms[0].y, y_max = local_atoms[0].y;
    double z_min = local_atoms[0].z, z_max = local_atoms[0].z;
    for (const auto& a : local_atoms)
    {
        x_min = std::min(x_min, a.x);
        x_max = std::max(x_max, a.x);
        y_min = std::min(y_min, a.y);
        y_max = std::max(y_max, a.y);
        z_min = std::min(z_min, a.z);
        z_max = std::max(z_max, a.z);
    }

    int nx = static_cast<int>(std::ceil((x_max - x_min) / grid_size)) + 1;
    int ny = static_cast<int>(std::ceil((y_max - y_min) / grid_size)) + 1;
    int nz = static_cast<int>(std::ceil((z_max - z_min) / grid_size)) + 1;

    std::vector<std::vector<std::vector<std::vector<int>>>> grid(
        nx, std::vector<std::vector<std::vector<int>>>(
            ny, std::vector<std::vector<int>>(nz)));

    for (int i = 0; i < natoms; i++)
    {
        int bx = static_cast<int>(std::floor((local_atoms[i].x - x_min) / grid_size));
        int by = static_cast<int>(std::floor((local_atoms[i].y - y_min) / grid_size));
        int bz = static_cast<int>(std::floor((local_atoms[i].z - z_min) / grid_size));
        bx = std::max(0, std::min(bx, nx - 1));
        by = std::max(0, std::min(by, ny - 1));
        bz = std::max(0, std::min(bz, nz - 1));
        grid[bx][by][bz].push_back(i);
    }

    int base = natoms / size;
    int remainder = natoms % size;
    int my_start = rank * base + std::min(rank, remainder);
    int my_end = my_start + base + (rank < remainder ? 1 : 0);

    int search_range = static_cast<int>(std::ceil(cutoff / grid_size));

    int my_total_neighbors = 0;

#pragma omp parallel
    {
        int local_count = 0;

#pragma omp for schedule(dynamic, 10) nowait
        for (int i = my_start; i < my_end; i++)
        {
            int bxi = static_cast<int>(std::floor((local_atoms[i].x - x_min) / grid_size));
            int byi = static_cast<int>(std::floor((local_atoms[i].y - y_min) / grid_size));
            int bzi = static_cast<int>(std::floor((local_atoms[i].z - z_min) / grid_size));
            bxi = std::max(0, std::min(bxi, nx - 1));
            byi = std::max(0, std::min(byi, ny - 1));
            bzi = std::max(0, std::min(bzi, nz - 1));

            for (int dx = -search_range; dx <= search_range; dx++)
            {
                for (int dy = -search_range; dy <= search_range; dy++)
                {
                    for (int dz = -search_range; dz <= search_range; dz++)
                    {
                        int bx = bxi + dx;
                        int by = byi + dy;
                        int bz = bzi + dz;
                        if (bx < 0 || bx >= nx || by < 0 || by >= ny || bz < 0 || bz >= nz)
                            continue;

                        for (int j : grid[bx][by][bz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = local_atoms[i].x - local_atoms[j].x;
                            double dy2 = local_atoms[i].y - local_atoms[j].y;
                            double dz2 = local_atoms[i].z - local_atoms[j].z;
                            double dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                            if (dist2 < cutoff2)
                            {
                                local_count++;
                            }
                        }
                    }
                }
            }
        }

#pragma omp atomic
        my_total_neighbors += local_count;
    }

    int global_total = 0;
    MPI_Reduce(&my_total_neighbors, &global_total, 1, MPI_INT, MPI_SUM, 0, comm);

    double mpi_end = MPI_Wtime();

    if (rank == 0)
    {
        total_neighbors = global_total;
    }
    return (mpi_end - mpi_start) * 1000.0;
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
         {{0, 0.0, 0.0, 0.0}, {0, 0.0, 0.5, 0.5}, {0, 0.5, 0.0, 0.5}, {0, 0.5, 0.5, 0.0}}, 1000, 12.0,
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
         2000, 4.0, "LCAO / semiconductor"},
        {"NaCl", 15, 5, 5, 2.4, 2.4, 2.4, 1.35, 2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.0, 0.5, 0.5},
          {0, 0.5, 0.0, 0.5},
          {0, 0.5, 0.5, 0.0},
          {1, 0.5, 0.0, 0.0},
          {1, 0.0, 0.5, 0.0},
          {1, 0.0, 0.0, 0.5},
          {1, 0.5, 0.5, 0.5}},
         3000, 6.0, "PW / ionic crystal"},
        {"TiO2 rutile", 10, 10, 7, 2.4, 2.4, 1.56, 1.25, 2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.5, 0.5, 0.5},
          {1, 0.305, 0.305, 0.0},
          {1, 0.695, 0.695, 0.0},
          {1, 0.805, 0.195, 0.5},
          {1, 0.195, 0.805, 0.5}},
         4200, 4.0, "LCAO / complex oxide"},
    };

    if (rank == 0)
    {
        std::cout << "\n========================================================================"
                  << std::endl;
        std::cout << "  ABACUS Material Neighbor Search - MPI+OpenMP Parallel Benchmark" << std::endl;
        std::cout << "========================================================================"
                  << std::endl;
        std::cout << "  MPI Processes:        " << size << std::endl;
        std::cout << "  OpenMP Threads/proc:  " << num_threads << std::endl;
        std::cout << "  Total parallel units: " << size * num_threads << std::endl;
        std::cout << "  Algorithm:            Grid-based neighbor search" << std::endl;
        std::cout << "========================================================================"
                  << std::endl;
    }

    struct BenchResult
    {
        std::string name;
        int natoms;
        double avg_nb;
        double serial_ms;
        double parallel_ms;
        double speedup;
        std::string category;
    };
    std::vector<BenchResult> results;

    for (size_t midx = 0; midx < materials.size(); ++midx)
    {
        const auto& mat = materials[midx];
        std::vector<AtomData> atoms;
        if (rank == 0)
        {
            atoms = generate_crystal_atoms(mat);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double serial_ms = 0.0;
        int serial_nb = 0;
        if (rank == 0)
        {
            serial_ms = run_grid_search_serial(atoms, mat.cutoff, serial_nb);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double parallel_ms = 0.0;
        int parallel_nb = 0;
        parallel_ms = run_grid_search_mpi_omp(atoms, mat.cutoff, MPI_COMM_WORLD, parallel_nb);

        if (rank == 0)
        {
            double avg_nb = static_cast<double>(serial_nb) / mat.expected_atoms;
            double speedup = (parallel_ms > 0.0) ? serial_ms / parallel_ms : 0.0;
            results.push_back(
                {mat.name, mat.expected_atoms, avg_nb, serial_ms, parallel_ms, speedup, mat.category});
        }
    }

    if (rank == 0)
    {
        std::cout << "\n";
        std::cout << std::left << std::setw(18) << "Material" << std::right << std::setw(8) << "Atoms"
                  << std::setw(8) << "AvgNB" << std::setw(13) << "Serial(ms)" << std::setw(13)
                  << "MPI+OMP(ms)" << std::setw(9) << "Speedup" << std::setw(11) << "Efficiency"
                  << std::setw(18) << "" << std::setw(22) << "Category" << std::endl;
        std::cout << std::string(110, '-') << std::endl;

        for (const auto& r : results)
        {
            std::cout << std::left << std::setw(18) << r.name << std::right << std::setw(8) << r.natoms
                      << std::setw(8) << std::fixed << std::setprecision(1) << r.avg_nb << std::setw(13)
                      << std::fixed << std::setprecision(2) << r.serial_ms << std::setw(13) << std::fixed
                      << std::setprecision(2) << r.parallel_ms << std::setw(8) << std::fixed
                      << std::setprecision(2) << r.speedup << "x" << std::setw(10) << std::fixed
                      << std::setprecision(1) << (r.speedup / (size * num_threads) * 100.0) << "%"
                      << std::setw(18) << "" << std::setw(22) << r.category << std::endl;
        }

        std::cout << std::string(110, '-') << std::endl;

        double total_s = 0, total_p = 0;
        for (const auto& r : results)
        {
            total_s += r.serial_ms;
            total_p += r.parallel_ms;
        }
        double overall = total_s / total_p;

        std::cout << std::left << std::setw(18) << "OVERALL" << std::right << std::setw(8) << ""
                  << std::setw(8) << "" << std::setw(13) << std::fixed << std::setprecision(2) << total_s
                  << std::setw(13) << std::fixed << std::setprecision(2) << total_p << std::setw(8)
                  << std::fixed << std::setprecision(2) << overall << "x" << std::setw(10) << std::fixed
                  << std::setprecision(1) << (overall / (size * num_threads) * 100.0) << "%" << std::endl;

        std::cout << "\n========================================================================"
                  << std::endl;
        std::cout << "  Configuration: " << size << " MPI x " << num_threads << " OpenMP = "
                  << size * num_threads << " parallel units" << std::endl;
        std::cout << "  Overall speedup:    " << std::fixed << std::setprecision(2) << overall << "x"
                  << std::endl;
        std::cout << "  Overall efficiency: " << std::fixed << std::setprecision(1)
                  << (overall / (size * num_threads) * 100.0) << "%" << std::endl;
        std::cout << "========================================================================"
                  << std::endl;

        std::ofstream csv("material_benchmark_results.csv");
        csv << "Material,Atoms,AvgNeighbors,Serial_ms,Parallel_ms,Speedup,MPI_Procs,OMP_Threads"
            << std::endl;
        for (const auto& r : results)
        {
            csv << r.name << "," << r.natoms << "," << r.avg_nb << "," << r.serial_ms << "," << r.parallel_ms
                << "," << r.speedup << "," << size << "," << num_threads << std::endl;
        }
        csv.close();
        std::cout << "  Results saved to: material_benchmark_results.csv" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
