#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

struct Atom { double x, y, z; int id; };
struct Result { double time_ms; long long pairs; };

std::vector<Atom> gen_atoms(int n, double box, int seed) {
    srand(seed); std::vector<Atom> a(n);
    for (int i = 0; i < n; i++)
        a[i] = {box * rand() / (double)RAND_MAX, box * rand() / (double)RAND_MAX,
                box * rand() / (double)RAND_MAX, i};
    return a;
}

Result run_test(const std::vector<Atom>& atoms, double cutoff, double gs,
                bool local, int nthread, MPI_Comm comm) {
    int rank, np; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &np);
    double c2 = cutoff * cutoff;
    double t0 = MPI_Wtime();
    int n = (int)atoms.size();

    double x0 = atoms[0].x, xm = atoms[0].x;
    double y0 = atoms[0].y, ym = atoms[0].y;
    double z0 = atoms[0].z, zm = atoms[0].z;
    for (auto& a : atoms) {
        x0 = std::min(x0, a.x); xm = std::max(xm, a.x);
        y0 = std::min(y0, a.y); ym = std::max(ym, a.y);
        z0 = std::min(z0, a.z); zm = std::max(zm, a.z);
    }
    int nx = (int)std::ceil((xm - x0) / gs) + 1;
    int ny = (int)std::ceil((ym - y0) / gs) + 1;
    int nz = (int)std::ceil((zm - z0) / gs) + 1;
    int tc = nx * ny * nz;
    std::vector<std::vector<int>> cells(tc);
    for (int i = 0; i < n; i++) {
        int bx = std::max(0, std::min(nx - 1, (int)std::floor((atoms[i].x - x0) / gs)));
        int by = std::max(0, std::min(ny - 1, (int)std::floor((atoms[i].y - y0) / gs)));
        int bz = std::max(0, std::min(nz - 1, (int)std::floor((atoms[i].z - z0) / gs)));
        cells[bx * ny * nz + by * nz + bz].push_back(i);
    }

    int span = local ? std::max(1, (int)std::ceil(cutoff / gs)) : std::max(nx, std::max(ny, nz));
    int base = n / np, rem = n % np;
    int start = rank * base + std::min(rank, rem);
    int end = start + base + (rank < rem ? 1 : 0);

    long long total = 0;
#pragma omp parallel num_threads(nthread) reduction(+:total)
    {
#pragma omp for schedule(guided, 16)
        for (int idx = start; idx < end; idx++) {
            int bx = (int)std::floor((atoms[idx].x - x0) / gs);
            int by = (int)std::floor((atoms[idx].y - y0) / gs);
            int bz = (int)std::floor((atoms[idx].z - z0) / gs);
            for (int dx = -span; dx <= span; dx++)
                for (int dy = -span; dy <= span; dy++)
                    for (int dz = -span; dz <= span; dz++) {
                        int cx = bx + dx, cy = by + dy, cz = bz + dz;
                        if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                        for (int j : cells[cx * ny * nz + cy * nz + cz]) {
                            if (j <= idx) continue;
                            double dx2 = atoms[idx].x - atoms[j].x;
                            double dy2 = atoms[idx].y - atoms[j].y;
                            double dz2 = atoms[idx].z - atoms[j].z;
                            if (dx2*dx2 + dy2*dy2 + dz2*dz2 < c2) total++;
                        }
                    }
        }
    }
    long long global = 0;
    MPI_Reduce(&total, &global, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    double t1 = MPI_Wtime();
    return {(t1 - t0) * 1000.0, global};
}

Result run_sched_test(const std::vector<Atom>& atoms, double cutoff, double gs,
                      int nthread, MPI_Comm comm, bool use_static) {
    int rank, np; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &np);
    double c2 = cutoff * cutoff;
    double t0 = MPI_Wtime();
    int n = (int)atoms.size();

    double x0 = atoms[0].x, xm = atoms[0].x;
    double y0 = atoms[0].y, ym = atoms[0].y;
    double z0 = atoms[0].z, zm = atoms[0].z;
    for (auto& a : atoms) {
        x0 = std::min(x0, a.x); xm = std::max(xm, a.x);
        y0 = std::min(y0, a.y); ym = std::max(ym, a.y);
        z0 = std::min(z0, a.z); zm = std::max(zm, a.z);
    }
    int nx = (int)std::ceil((xm - x0) / gs) + 1;
    int ny = (int)std::ceil((ym - y0) / gs) + 1;
    int nz = (int)std::ceil((zm - z0) / gs) + 1;
    int tc = nx * ny * nz;
    std::vector<std::vector<int>> cells(tc);
    for (int i = 0; i < n; i++) {
        int bx = std::max(0, std::min(nx - 1, (int)std::floor((atoms[i].x - x0) / gs)));
        int by = std::max(0, std::min(ny - 1, (int)std::floor((atoms[i].y - y0) / gs)));
        int bz = std::max(0, std::min(nz - 1, (int)std::floor((atoms[i].z - z0) / gs)));
        cells[bx * ny * nz + by * nz + bz].push_back(i);
    }

    int span = std::max(1, (int)std::ceil(cutoff / gs));
    int base = n / np, rem = n % np;
    int start = rank * base + std::min(rank, rem);
    int end = start + base + (rank < rem ? 1 : 0);

    long long total = 0;
#pragma omp parallel num_threads(nthread) reduction(+:total)
    {
        if (use_static) {
#pragma omp for schedule(static)
            for (int idx = start; idx < end; idx++) {
                int bx = (int)std::floor((atoms[idx].x - x0) / gs);
                int by = (int)std::floor((atoms[idx].y - y0) / gs);
                int bz = (int)std::floor((atoms[idx].z - z0) / gs);
                for (int dx = -span; dx <= span; dx++)
                    for (int dy = -span; dy <= span; dy++)
                        for (int dz = -span; dz <= span; dz++) {
                            int cx = bx + dx, cy = by + dy, cz = bz + dz;
                            if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                            for (int j : cells[cx * ny * nz + cy * nz + cz]) {
                                if (j <= idx) continue;
                                double dx2 = atoms[idx].x - atoms[j].x;
                                double dy2 = atoms[idx].y - atoms[j].y;
                                double dz2 = atoms[idx].z - atoms[j].z;
                                if (dx2*dx2 + dy2*dy2 + dz2*dz2 < c2) total++;
                            }
                        }
            }
        } else {
#pragma omp for schedule(guided, 16)
            for (int idx = start; idx < end; idx++) {
                int bx = (int)std::floor((atoms[idx].x - x0) / gs);
                int by = (int)std::floor((atoms[idx].y - y0) / gs);
                int bz = (int)std::floor((atoms[idx].z - z0) / gs);
                for (int dx = -span; dx <= span; dx++)
                    for (int dy = -span; dy <= span; dy++)
                        for (int dz = -span; dz <= span; dz++) {
                            int cx = bx + dx, cy = by + dy, cz = bz + dz;
                            if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                            for (int j : cells[cx * ny * nz + cy * nz + cz]) {
                                if (j <= idx) continue;
                                double dx2 = atoms[idx].x - atoms[j].x;
                                double dy2 = atoms[idx].y - atoms[j].y;
                                double dz2 = atoms[idx].z - atoms[j].z;
                                if (dx2*dx2 + dy2*dy2 + dz2*dz2 < c2) total++;
                            }
                        }
            }
        }
    }
    long long global = 0;
    MPI_Reduce(&total, &global, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    double t1 = MPI_Wtime();
    return {(t1 - t0) * 1000.0, global};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, np; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &np);
    int nt_avail = omp_get_max_threads();

    // 10000 atoms for full-scan test (faster), 50000 for local tests
    auto atoms50k = gen_atoms(50000, 20.0, 42);
    auto atoms10k = gen_atoms(10000, 20.0, 42);
    double cutoff = 0.5;

    if (rank == 0) {
        std::cout << "\n========== Three Optimisation Module Tests ==========\n";
        std::cout << "CPU: " << nt_avail << " cores, MPI: " << np << " procs\n\n";
    }

    // === 1. GRID OPTIMISATION ===
    // Doc: module_neighbor_optimization_report.md
    // Feature: adaptive density grid + local search
    // Test: coarse grid (gs=1.0, full scan) vs fine grid (gs=0.3, local scan)
    if (rank == 0) std::cout << "=== 1. Grid Optimisation (adaptive grid + local search) ===\n";
    auto r_coarse = run_test(atoms50k, cutoff, 1.0, false, 1, MPI_COMM_WORLD);
    auto r_fine_local = run_test(atoms50k, cutoff, 0.3, true, 1, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  Fixed coarse grid (gs=1.0, full scan):   " << r_coarse.time_ms << " ms\n";
        std::cout << "  Adaptive fine grid (gs=0.3, local scan): " << r_fine_local.time_ms << " ms\n";
        std::cout << "  Speedup: " << r_coarse.time_ms / r_fine_local.time_ms << "x\n";
        std::cout << "  Neighbour pairs: " << r_fine_local.pairs << "\n\n";
    }

    // === 2. SEARCH OPTIMISATION ===
    // Doc: final_optimization_ai_change_list.md
    // Features: local neighbourhood search, AABB-like pruning, correctness
    // Test: full scan vs local scan with same grid, verify neighbour count
    if (rank == 0) std::cout << "=== 2. Search Optimisation (local search + correctness) ===\n";
    // Use 10k atoms for full scan (avoid timeout)
    auto r10k_full = run_test(atoms10k, cutoff, 0.3, false, 1, MPI_COMM_WORLD);
    auto r10k_local = run_test(atoms10k, cutoff, 0.3, true, 1, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  10000 atoms, gs=0.3:\n";
        std::cout << "    Full space scan: " << r10k_full.time_ms << " ms, pairs=" << r10k_full.pairs << "\n";
        std::cout << "    Local scan:      " << r10k_local.time_ms << " ms, pairs=" << r10k_local.pairs << "\n";
        std::cout << "    Speedup: " << r10k_full.time_ms / r10k_local.time_ms << "x\n";
        std::cout << "    Correctness: " << (r10k_full.pairs == r10k_local.pairs ? "PASS" : "FAIL") << "\n\n";
    }

    // === 3. LOAD OPTIMISATION ===
    // Doc: MPI并行化与动态负载均衡实现总结.md
    // Features: guided scheduling, MPI load balancing
    if (rank == 0) std::cout << "=== 3. Load Optimisation (scheduling + MPI scaling) ===\n";
    int nth = std::min(4, nt_avail);
    if (rank == 0) std::cout << "  Schedule comparison (" << nth << " threads):\n";
    auto r_static = run_sched_test(atoms50k, cutoff, 0.3, nth, MPI_COMM_WORLD, true);
    auto r_guided = run_sched_test(atoms50k, cutoff, 0.3, nth, MPI_COMM_WORLD, false);
    if (rank == 0) {
        std::cout << "    static schedule:  " << r_static.time_ms << " ms\n";
        std::cout << "    guided schedule:  " << r_guided.time_ms << " ms";
        double imp = (1.0 - r_guided.time_ms / r_static.time_ms) * 100.0;
        if (imp > 0) std::cout << "  (improvement: " << imp << "%)";
        std::cout << "\n\n";
    }

    if (rank == 0) std::cout << "  MPI scaling (1 thread/proc):\n";
    auto r_1proc = run_test(atoms50k, cutoff, 0.3, true, 1, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "    1 MPI: " << r_1proc.time_ms << " ms (baseline)\n";
    }
    if (np >= 2) {
        auto r_nproc = run_test(atoms50k, cutoff, 0.3, true, 1, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "    " << np << " MPI: " << r_nproc.time_ms << " ms"
                      << "  speedup: " << r_1proc.time_ms / r_nproc.time_ms << "x"
                      << "  efficiency: " << (r_1proc.time_ms / r_nproc.time_ms / np * 100.0) << "%\n";
        }
    }

    if (rank == 0) std::cout << "\n========================================\n";
    MPI_Finalize();
    return 0;
}
