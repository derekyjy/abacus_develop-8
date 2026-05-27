#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct AtomData { double x, y, z; int type, id; };

struct Mate
{
    std::string name;
    int nx, ny, nz;
    double lx, ly, lz, cutoff;
    int ntype, expected_atoms;
    std::vector<std::tuple<int, double, double, double>> basis;
};

std::vector<AtomData> gen_crystal(const Mate& m)
{
    std::vector<AtomData> a;
    a.reserve(m.expected_atoms);
    int id = 0;
    for (int ix = 0; ix < m.nx; ++ix)
        for (int iy = 0; iy < m.ny; ++iy)
            for (int iz = 0; iz < m.nz; ++iz)
                for (auto& b : m.basis)
                {
                    if (id >= m.expected_atoms)
                        break;
                    a.push_back({(ix + std::get<1>(b)) * m.lx, (iy + std::get<2>(b)) * m.ly,
                                 (iz + std::get<3>(b)) * m.lz, std::get<0>(b), id++});
                }
    return a;
}

double run_serial(const std::vector<AtomData>& a, double cut, int& nb)
{
    double t0 = omp_get_wtime();
    double gs = cut + 0.1, c2 = cut * cut, cnt = 0;
    int n = (int)a.size();

    double x0 = a[0].x, y0 = a[0].y, z0 = a[0].z;
    double xm = a[0].x, ym = a[0].y, zm = a[0].z;
    for (auto& t : a)
    {
        x0 = std::min(x0, t.x); xm = std::max(xm, t.x);
        y0 = std::min(y0, t.y); ym = std::max(ym, t.y);
        z0 = std::min(z0, t.z); zm = std::max(zm, t.z);
    }
    int nx = (int)std::ceil((xm - x0) / gs) + 1;
    int ny = (int)std::ceil((ym - y0) / gs) + 1;
    int nz = (int)std::ceil((zm - z0) / gs) + 1;

    std::vector<std::vector<std::vector<std::vector<int>>>> g(
        nx, std::vector<std::vector<std::vector<int>>>(ny, std::vector<std::vector<int>>(nz)));
    for (int i = 0; i < n; i++)
    {
        int bx = std::max(0, std::min(nx - 1, (int)std::floor((a[i].x - x0) / gs)));
        int by = std::max(0, std::min(ny - 1, (int)std::floor((a[i].y - y0) / gs)));
        int bz = std::max(0, std::min(nz - 1, (int)std::floor((a[i].z - z0) / gs)));
        g[bx][by][bz].push_back(i);
    }

    for (int i = 0; i < n; i++)
        for (int bx = 0; bx < nx; bx++)
            for (int by = 0; by < ny; by++)
                for (int bz = 0; bz < nz; bz++)
                    for (int j : g[bx][by][bz])
                    {
                        if (j <= i)
                            continue;
                        double dx = a[i].x - a[j].x;
                        double dy = a[i].y - a[j].y;
                        double dz = a[i].z - a[j].z;
                        if (dx * dx + dy * dy + dz * dz < c2)
                            cnt += 1.0;
                    }
    nb = (int)cnt;
    return (omp_get_wtime() - t0) * 1000.0;
}

double run_parallel(const std::vector<AtomData>& a, double cut, MPI_Comm comm, int& nb)
{
    int rk, np;
    MPI_Comm_rank(comm, &rk);
    MPI_Comm_size(comm, &np);
    double t0 = MPI_Wtime();
    double gs = cut + 0.1, c2 = cut * cut;
    int n = (int)a.size();

    double x0 = a[0].x, y0 = a[0].y, z0 = a[0].z;
    double xm = a[0].x, ym = a[0].y, zm = a[0].z;
    for (auto& t : a)
    {
        x0 = std::min(x0, t.x); xm = std::max(xm, t.x);
        y0 = std::min(y0, t.y); ym = std::max(ym, t.y);
        z0 = std::min(z0, t.z); zm = std::max(zm, t.z);
    }
    int nx = (int)std::ceil((xm - x0) / gs) + 1;
    int ny = (int)std::ceil((ym - y0) / gs) + 1;
    int nz = (int)std::ceil((zm - z0) / gs) + 1;

    std::vector<std::vector<std::vector<std::vector<int>>>> g(
        nx, std::vector<std::vector<std::vector<int>>>(ny, std::vector<std::vector<int>>(nz)));
    for (int i = 0; i < n; i++)
    {
        int bx = std::max(0, std::min(nx - 1, (int)std::floor((a[i].x - x0) / gs)));
        int by = std::max(0, std::min(ny - 1, (int)std::floor((a[i].y - y0) / gs)));
        int bz = std::max(0, std::min(nz - 1, (int)std::floor((a[i].z - z0) / gs)));
        g[bx][by][bz].push_back(i);
    }

    int base = n / np, rem = n % np;
    int start = rk * base + std::min(rk, rem);
    int end = start + base + (rk < rem ? 1 : 0);

    double cnt = 0;
#pragma omp parallel
    {
        double lc = 0;
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            for (int bx = 0; bx < nx; bx++)
                for (int by = 0; by < ny; by++)
                    for (int bz = 0; bz < nz; bz++)
                        for (int j : g[bx][by][bz])
                        {
                            if (j <= i)
                                continue;
                            double dx = a[i].x - a[j].x;
                            double dy = a[i].y - a[j].y;
                            double dz = a[i].z - a[j].z;
                            if (dx * dx + dy * dy + dz * dz < c2)
                                lc += 1.0;
                        }
        }
#pragma omp atomic
        cnt += lc;
    }

    int my = (int)cnt;
    MPI_Request req;
    MPI_Iallreduce(&my, &nb, 1, MPI_INT, MPI_SUM, comm, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    return (MPI_Wtime() - t0) * 1000.0;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rk, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int nt = 1;
#pragma omp parallel
#pragma omp master
    nt = omp_get_num_threads();

    std::vector<Mate> mats = {
        {"Al fcc", 20, 10, 10, 2.0, 2.0, 2.0, 1.5, 1, 8000,
         {{0, 0.0, 0.0, 0.0}, {0, 0.0, 0.5, 0.5}, {0, 0.5, 0.0, 0.5}, {0, 0.5, 0.5, 0.0}}},
        {"Si diamond", 20, 10, 10, 2.4, 2.4, 2.4, 1.25, 1, 16000,
         {{0, 0.0, 0.0, 0.0},  {0, 0.0, 0.5, 0.5},  {0, 0.5, 0.0, 0.5},  {0, 0.5, 0.5, 0.0},
          {0, 0.25, 0.25, 0.25}, {0, 0.25, 0.75, 0.75}, {0, 0.75, 0.25, 0.75}, {0, 0.75, 0.75, 0.25}}},
        {"NaCl", 30, 10, 10, 2.4, 2.4, 2.4, 1.35, 2, 24000,
         {{0, 0.0, 0.0, 0.0}, {0, 0.0, 0.5, 0.5}, {0, 0.5, 0.0, 0.5}, {0, 0.5, 0.5, 0.0},
          {1, 0.5, 0.0, 0.0}, {1, 0.0, 0.5, 0.0}, {1, 0.0, 0.0, 0.5}, {1, 0.5, 0.5, 0.5}}},
        {"TiO2 rutile", 20, 20, 14, 2.4, 2.4, 1.56, 1.25, 2, 33600,
         {{0, 0.0, 0.0, 0.0},   {0, 0.5, 0.5, 0.5},   {1, 0.305, 0.305, 0.0},
          {1, 0.695, 0.695, 0.0}, {1, 0.805, 0.195, 0.5}, {1, 0.195, 0.805, 0.5}}},
    };

    if (rk == 0)
    {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  ABACUS Neighbor Search - MPI+OpenMP Benchmark" << std::endl;
        std::cout << "  Cores: " << np * nt << " (" << np << " MPI x " << nt << " OMP)" << std::endl;
        std::cout << "  Algo: Full-traversal | Crystals: 8x scaled" << std::endl;
        std::cout << "  Comm: MPI_Iallreduce (non-blocking)" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }

    struct R { std::string nm; int na; double ser, par, sp, eff; };
    std::vector<R> rs;

    for (auto& m : mats)
    {
        std::vector<AtomData> at = gen_crystal(m);

        double ser = 0; int snb = 0;
        if (rk == 0) ser = run_serial(at, m.cutoff, snb);

        MPI_Barrier(MPI_COMM_WORLD);

        double par = 0; int pnb = 0;
        par = run_parallel(at, m.cutoff, MPI_COMM_WORLD, pnb);

        if (rk == 0)
        {
            double u = np * nt;
            rs.push_back({m.name, (int)at.size(), ser, par, ser / par, ser / par / u * 100.0});
        }
    }

    if (rk == 0)
    {
        std::cout << "  " << std::left << std::setw(14) << "Case" << std::right << std::setw(8) << "Atoms"
                  << std::setw(12) << "Ser(ms)" << std::setw(12) << "Par(ms)" << std::setw(8) << "Speedup"
                  << std::setw(9) << "Eff%" << std::endl;
        std::cout << "  " << std::string(63, '-') << std::endl;
        double ts = 0, tp = 0;
        for (auto& r : rs)
        {
            ts += r.ser; tp += r.par;
            std::cout << "  " << std::left << std::setw(14) << r.nm << std::right << std::setw(8) << r.na
                      << std::setw(12) << std::fixed << std::setprecision(0) << r.ser
                      << std::setw(12) << r.par << std::setw(7) << std::setprecision(1) << r.sp << "x"
                      << std::setw(8) << std::setprecision(1) << r.eff << "%" << std::endl;
        }
        std::cout << "  " << std::string(63, '-') << std::endl;
        double u = np * nt, ov = ts / tp;
        std::cout << "  " << std::left << std::setw(14) << "OVERALL" << std::right << std::setw(8) << ""
                  << std::setw(12) << std::fixed << std::setprecision(0) << ts << std::setw(12) << tp
                  << std::setw(7) << std::setprecision(1) << ov << "x" << std::setw(8)
                  << std::setprecision(1) << (ov / u * 100.0) << "%" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
