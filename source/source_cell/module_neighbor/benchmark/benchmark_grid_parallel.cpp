#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct Atom3D
{
    double x, y, z;
    int type;
    int id;
};

struct NeighborEntry
{
    int type;
    int id;
    double x, y, z;
};

struct SearchResult
{
    double serial_time;
    double parallel_time;
    double speedup;
    int total_neighbors;
    int natoms;
    int mpi_procs;
    int omp_threads;
};

const double CUTOFF_RADIUS = 5.0;
const double BOX_SIZE = 100.0;

std::vector<Atom3D> generate_random_atoms(int n)
{
    std::srand(static_cast<unsigned int>(42));
    std::vector<Atom3D> atoms(n);
    for (int i = 0; i < n; i++)
    {
        atoms[i].x = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].y = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].z = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].type = 0;
        atoms[i].id = i;
    }
    return atoms;
}

double grid_search_serial(const std::vector<Atom3D>& atoms)
{
    double start = omp_get_wtime();

    double cutoff2 = CUTOFF_RADIUS * CUTOFF_RADIUS;
    double grid_size = CUTOFF_RADIUS + 0.1;

    double x_min = BOX_SIZE, x_max = 0;
    double y_min = BOX_SIZE, y_max = 0;
    double z_min = BOX_SIZE, z_max = 0;
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

    auto get_bin = [&](double x, double y, double z) -> std::tuple<int, int, int> {
        int bx = static_cast<int>(std::floor((x - x_min) / grid_size));
        int by = static_cast<int>(std::floor((y - y_min) / grid_size));
        int bz = static_cast<int>(std::floor((z - z_min) / grid_size));
        bx = std::max(0, std::min(bx, nx - 1));
        by = std::max(0, std::min(by, ny - 1));
        bz = std::max(0, std::min(bz, nz - 1));
        return {bx, by, bz};
    };

    for (size_t i = 0; i < atoms.size(); i++)
    {
        auto [bx, by, bz] = get_bin(atoms[i].x, atoms[i].y, atoms[i].z);
        grid[bx][by][bz].push_back(static_cast<int>(i));
    }

    std::vector<std::vector<int>> neighbors(atoms.size());
    int search_range = static_cast<int>(std::ceil(CUTOFF_RADIUS / grid_size));

    for (size_t i = 0; i < atoms.size(); i++)
    {
        auto [bxi, byi, bzi] = get_bin(atoms[i].x, atoms[i].y, atoms[i].z);

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
                        if (static_cast<size_t>(j) <= i)
                            continue;
                        double dx2 = atoms[i].x - atoms[j].x;
                        double dy2 = atoms[i].y - atoms[j].y;
                        double dz2 = atoms[i].z - atoms[j].z;
                        double dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                        if (dist2 < cutoff2)
                        {
                            neighbors[i].push_back(j);
                            neighbors[j].push_back(static_cast<int>(i));
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    return end - start;
}

double grid_search_mpi_omp(const std::vector<Atom3D>& atoms, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double start = MPI_Wtime();

    int natoms = static_cast<int>(atoms.size());
    MPI_Bcast(&natoms, 1, MPI_INT, 0, comm);

    std::vector<double> atom_coords;
    if (rank == 0)
    {
        atom_coords.resize(natoms * 4);
        for (int i = 0; i < natoms; i++)
        {
            atom_coords[i * 4] = atoms[i].x;
            atom_coords[i * 4 + 1] = atoms[i].y;
            atom_coords[i * 4 + 2] = atoms[i].z;
            atom_coords[i * 4 + 3] = static_cast<double>(atoms[i].id);
        }
    }
    else
    {
        atom_coords.resize(natoms * 4);
    }
    MPI_Bcast(atom_coords.data(), natoms * 4, MPI_DOUBLE, 0, comm);

    std::vector<Atom3D> local_atoms;
    if (rank != 0)
    {
        local_atoms.resize(natoms);
        for (int i = 0; i < natoms; i++)
        {
            local_atoms[i].x = atom_coords[i * 4];
            local_atoms[i].y = atom_coords[i * 4 + 1];
            local_atoms[i].z = atom_coords[i * 4 + 2];
            local_atoms[i].id = static_cast<int>(atom_coords[i * 4 + 3]);
            local_atoms[i].type = 0;
        }
    }
    const std::vector<Atom3D>& work_atoms = (rank == 0) ? atoms : local_atoms;

    double cutoff2 = CUTOFF_RADIUS * CUTOFF_RADIUS;
    double grid_size = CUTOFF_RADIUS + 0.1;

    double x_min = BOX_SIZE, x_max = 0;
    double y_min = BOX_SIZE, y_max = 0;
    double z_min = BOX_SIZE, z_max = 0;
    for (const auto& a : work_atoms)
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
        int bx = static_cast<int>(std::floor((work_atoms[i].x - x_min) / grid_size));
        int by = static_cast<int>(std::floor((work_atoms[i].y - y_min) / grid_size));
        int bz = static_cast<int>(std::floor((work_atoms[i].z - z_min) / grid_size));
        bx = std::max(0, std::min(bx, nx - 1));
        by = std::max(0, std::min(by, ny - 1));
        bz = std::max(0, std::min(bz, nz - 1));
        grid[bx][by][bz].push_back(i);
    }

    int base = natoms / size;
    int remainder = natoms % size;
    int my_start = rank * base + std::min(rank, remainder);
    int my_end = my_start + base + (rank < remainder ? 1 : 0);
    int my_count = my_end - my_start;

    int search_range = static_cast<int>(std::ceil(CUTOFF_RADIUS / grid_size));

    struct PairData
    {
        int i;
        int j;
    };
    std::vector<PairData> my_pairs;

#pragma omp parallel
    {
        std::vector<PairData> local_pairs;

#pragma omp for schedule(dynamic, 10) nowait
        for (int idx = my_start; idx < my_end; idx++)
        {
            int i = idx;
            int bxi = static_cast<int>(std::floor((work_atoms[i].x - x_min) / grid_size));
            int byi = static_cast<int>(std::floor((work_atoms[i].y - y_min) / grid_size));
            int bzi = static_cast<int>(std::floor((work_atoms[i].z - z_min) / grid_size));
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
                            double dx2 = work_atoms[i].x - work_atoms[j].x;
                            double dy2 = work_atoms[i].y - work_atoms[j].y;
                            double dz2 = work_atoms[i].z - work_atoms[j].z;
                            double dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                            if (dist2 < cutoff2)
                            {
                                local_pairs.push_back({i, j});
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            my_pairs.insert(my_pairs.end(), local_pairs.begin(), local_pairs.end());
        }
    }

    int my_pair_count = static_cast<int>(my_pairs.size());
    std::vector<int> all_counts(size);
    MPI_Allgather(&my_pair_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

    int total_pairs = 0;
    std::vector<int> displs(size, 0);
    for (int i = 0; i < size; i++)
    {
        displs[i] = total_pairs;
        total_pairs += all_counts[i];
    }

    std::vector<int> gather_i, gather_j;
    if (rank == 0)
    {
        gather_i.resize(total_pairs);
        gather_j.resize(total_pairs);
    }

    std::vector<int> my_i(my_pair_count), my_j(my_pair_count);
    for (int p = 0; p < my_pair_count; p++)
    {
        my_i[p] = my_pairs[p].i;
        my_j[p] = my_pairs[p].j;
    }

    MPI_Gatherv(my_i.data(), my_pair_count, MPI_INT,
                rank == 0 ? gather_i.data() : nullptr,
                all_counts.data(), displs.data(), MPI_INT, 0, comm);
    MPI_Gatherv(my_j.data(), my_pair_count, MPI_INT,
                rank == 0 ? gather_j.data() : nullptr,
                all_counts.data(), displs.data(), MPI_INT, 0, comm);

    double end = MPI_Wtime();
    return end - start;
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

    if (rank == 0)
    {
        std::cout << "=================================================================" << std::endl;
        std::cout << "  ABACUS Grid-Based Neighbor Search - MPI+OpenMP Benchmark" << std::endl;
        std::cout << "=================================================================" << std::endl;
        std::cout << "  MPI Processes:      " << size << std::endl;
        std::cout << "  OpenMP Threads/proc:" << num_threads << std::endl;
        std::cout << "  Cutoff Radius:      " << CUTOFF_RADIUS << std::endl;
        std::cout << "  Box Size:           " << BOX_SIZE << "^3" << std::endl;
        std::cout << "  Algorithm:          Grid-based O(N)" << std::endl;
        std::cout << "=================================================================" << std::endl;
    }

    std::vector<int> test_sizes = {500, 1000, 2000, 5000, 10000, 20000, 50000};

    if (rank == 0)
    {
        std::cout << std::endl;
        std::cout << std::setw(10) << "Atoms" << std::setw(14) << "Serial(s)" << std::setw(14) << "MPI+OMP(s)"
                  << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
    }

    std::vector<SearchResult> results;

    for (int natoms : test_sizes)
    {
        std::vector<Atom3D> atoms;
        if (rank == 0)
        {
            atoms = generate_random_atoms(natoms);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double serial_time = 0.0;
        if (rank == 0)
        {
            serial_time = grid_search_serial(atoms);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double parallel_time = grid_search_mpi_omp(atoms, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double speedup = (parallel_time > 0.0) ? serial_time / parallel_time : 0.0;
            double efficiency = speedup / (size * num_threads) * 100.0;

            std::cout << std::setw(10) << natoms << std::setw(14) << std::fixed << std::setprecision(6)
                      << serial_time << std::setw(14) << std::fixed << std::setprecision(6) << parallel_time
                      << std::setw(9) << std::fixed << std::setprecision(2) << speedup << "x" << std::setw(11)
                      << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;

            SearchResult res;
            res.natoms = natoms;
            res.serial_time = serial_time;
            res.parallel_time = parallel_time;
            res.speedup = speedup;
            res.total_neighbors = 0;
            res.mpi_procs = size;
            res.omp_threads = num_threads;
            results.push_back(res);
        }
    }

    if (rank == 0)
    {
        std::ofstream csv("grid_benchmark_results.csv");
        csv << "Atoms,Serial_Time_s,MPI_Time_s,Speedup,MPI_Procs,OMP_Threads" << std::endl;
        for (const auto& r : results)
        {
            csv << r.natoms << "," << r.serial_time << "," << r.parallel_time << "," << r.speedup << ","
                << r.mpi_procs << "," << r.omp_threads << std::endl;
        }
        csv.close();

        std::cout << std::endl;
        std::cout << "=================================================================" << std::endl;
        std::cout << "  Summary" << std::endl;
        std::cout << "=================================================================" << std::endl;
        std::cout << "  Configuration: " << size << " MPI x " << num_threads << " OpenMP threads" << std::endl;

        double avg_speedup = 0;
        for (const auto& r : results)
        {
            avg_speedup += r.speedup;
        }
        avg_speedup /= results.size();
        std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
        std::cout << "  Average Efficiency: " << std::fixed << std::setprecision(1)
                  << (avg_speedup / (size * num_threads) * 100.0) << "%" << std::endl;
        std::cout << "  Results saved to: grid_benchmark_results.csv" << std::endl;
        std::cout << "=================================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
