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

const double CUTOFF_RADIUS = 5.0;
const double BOX_SIZE = 50.0;

struct Atom
{
    double x, y, z;
    int id;
};

struct NeighborPair
{
    int atom_i;
    int atom_j;
};

std::vector<double> serial_neighbor_search(const std::vector<Atom>& atoms, int& total_neighbors)
{
    double start = omp_get_wtime();

    std::vector<std::vector<int>> neighbors(atoms.size());
    total_neighbors = 0;
    double cutoff2 = CUTOFF_RADIUS * CUTOFF_RADIUS;

    for (size_t i = 0; i < atoms.size(); i++)
    {
        for (size_t j = i + 1; j < atoms.size(); j++)
        {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            double dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < cutoff2)
            {
                neighbors[i].push_back(static_cast<int>(j));
                neighbors[j].push_back(static_cast<int>(i));
                total_neighbors++;
            }
        }
    }

    double end = omp_get_wtime();
    return {end - start};
}

std::vector<double> mpi_openmp_neighbor_search(const std::vector<Atom>& atoms,
                                               MPI_Comm comm,
                                               int& total_neighbors)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double start = MPI_Wtime();

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

    std::vector<Atom> local_atoms;
    if (rank != 0)
    {
        local_atoms.resize(natoms);
        for (int i = 0; i < natoms; i++)
        {
            local_atoms[i].x = atom_data[i * 3];
            local_atoms[i].y = atom_data[i * 3 + 1];
            local_atoms[i].z = atom_data[i * 3 + 2];
            local_atoms[i].id = i;
        }
    }

    const std::vector<Atom>& work_atoms = (rank == 0) ? atoms : local_atoms;

    int base = natoms / size;
    int remainder = natoms % size;
    int my_start = rank * base + std::min(rank, remainder);
    int my_end = my_start + base + (rank < remainder ? 1 : 0);

    std::vector<int> recv_counts(size);
    int local_count = my_end - my_start;
    MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> recv_displs(size, 0);
    for (int i = 1; i < size; i++)
    {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    double cutoff2 = CUTOFF_RADIUS * CUTOFF_RADIUS;

    std::vector<int> my_neighbors_i;
    std::vector<int> my_neighbors_j;

#pragma omp parallel
    {
        std::vector<int> local_ni;
        std::vector<int> local_nj;

#pragma omp for schedule(dynamic, 10) nowait
        for (int i = my_start; i < my_end; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                if (i == j)
                    continue;
                double dx = work_atoms[i].x - work_atoms[j].x;
                double dy = work_atoms[i].y - work_atoms[j].y;
                double dz = work_atoms[i].z - work_atoms[j].z;
                double dist2 = dx * dx + dy * dy + dz * dz;
                if (dist2 < cutoff2)
                {
                    local_ni.push_back(i);
                    local_nj.push_back(j);
                }
            }
        }

#pragma omp critical
        {
            my_neighbors_i.insert(my_neighbors_i.end(), local_ni.begin(), local_ni.end());
            my_neighbors_j.insert(my_neighbors_j.end(), local_nj.begin(), local_nj.end());
        }
    }

    int my_pair_count = static_cast<int>(my_neighbors_i.size());
    std::vector<int> all_pair_counts(size);
    MPI_Allgather(&my_pair_count, 1, MPI_INT, all_pair_counts.data(), 1, MPI_INT, comm);

    int total_pairs = 0;
    std::vector<int> all_displs(size, 0);
    for (int i = 0; i < size; i++)
    {
        all_displs[i] = total_pairs;
        total_pairs += all_pair_counts[i];
    }

    std::vector<int> gathered_i, gathered_j;
    if (rank == 0)
    {
        gathered_i.resize(total_pairs);
        gathered_j.resize(total_pairs);
    }

    MPI_Gatherv(my_neighbors_i.data(), my_pair_count, MPI_INT,
                rank == 0 ? gathered_i.data() : nullptr,
                all_pair_counts.data(), all_displs.data(), MPI_INT, 0, comm);

    MPI_Gatherv(my_neighbors_j.data(), my_pair_count, MPI_INT,
                rank == 0 ? gathered_j.data() : nullptr,
                all_pair_counts.data(), all_displs.data(), MPI_INT, 0, comm);

    double end = MPI_Wtime();

    double comp_time = end - start;

    if (rank == 0)
    {
        total_neighbors = total_pairs / 2;
    }

    return {comp_time};
}

std::vector<Atom> generate_random_atoms(int n)
{
    std::srand(42);
    std::vector<Atom> atoms(n);
    for (int i = 0; i < n; i++)
    {
        atoms[i].x = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].y = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].z = BOX_SIZE * static_cast<double>(std::rand()) / RAND_MAX;
        atoms[i].id = i;
    }
    return atoms;
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
        std::cout << "============================================================" << std::endl;
        std::cout << "  ABACUS Neighbor Atom Search - MPI+OpenMP Benchmark" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "  MPI Processes: " << size << std::endl;
        std::cout << "  OpenMP Threads per process: " << num_threads << std::endl;
        std::cout << "  Cutoff Radius: " << CUTOFF_RADIUS << std::endl;
        std::cout << "  Box Size: " << BOX_SIZE << " x " << BOX_SIZE << " x " << BOX_SIZE << std::endl;
        std::cout << "============================================================" << std::endl;
    }

    std::vector<int> test_sizes = {500, 1000, 2000, 5000, 10000};

    std::ofstream result_file;
    if (rank == 0)
    {
        result_file.open("benchmark_results.csv");
        result_file << "Atoms,Serial_Time_s,MPI_Time_s,Speedup,Neighbors,MPI_Procs,OMP_Threads" << std::endl;
    }

    for (int natoms : test_sizes)
    {
        std::vector<Atom> atoms;
        if (rank == 0)
        {
            atoms = generate_random_atoms(natoms);
        }

        int total_neighbors_serial = 0;
        double serial_time = 0.0;
        if (rank == 0)
        {
            auto result = serial_neighbor_search(atoms, total_neighbors_serial);
            serial_time = result[0];
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int total_neighbors_parallel = 0;
        auto par_result = mpi_openmp_neighbor_search(atoms, MPI_COMM_WORLD, total_neighbors_parallel);
        double parallel_time = par_result[0];

        double speedup = 0.0;
        if (rank == 0)
        {
            speedup = (parallel_time > 0.0) ? serial_time / parallel_time : 0.0;

            std::cout << std::endl;
            std::cout << "--- Test with " << natoms << " atoms ---" << std::endl;
            std::cout << "  Serial time:       " << std::fixed << std::setprecision(6) << serial_time
                      << " s" << std::endl;
            std::cout << "  MPI+OpenMP time:   " << std::fixed << std::setprecision(6) << parallel_time
                      << " s" << std::endl;
            std::cout << "  Speedup:           " << std::fixed << std::setprecision(3) << speedup << "x"
                      << std::endl;
            std::cout << "  Efficiency:        " << std::fixed << std::setprecision(1)
                      << (speedup / (size * num_threads) * 100.0) << "%" << std::endl;
            std::cout << "  Neighbors found:   " << total_neighbors_serial << " (serial), "
                      << total_neighbors_parallel << " (parallel)" << std::endl;

            result_file << natoms << "," << serial_time << "," << parallel_time << "," << speedup << ","
                        << total_neighbors_serial << "," << size << "," << num_threads << std::endl;
        }
    }

    if (rank == 0)
    {
        result_file.close();
        std::cout << std::endl;
        std::cout << "Results written to benchmark_results.csv" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
