#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct AtomData { double x, y, z; int type, id; };

struct MaterialCase
{
    std::string name;
    int nx, ny, nz;
    double lx, ly, lz, cutoff;
    int ntype, expected_atoms;
    std::vector<std::tuple<int, double, double, double>> basis;
    std::string category;
};

std::vector<AtomData> gen_crystal(const MaterialCase& m)
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
                    a.push_back({(ix + std::get<1>(b)) * m.lx,
                                 (iy + std::get<2>(b)) * m.ly,
                                 (iz + std::get<3>(b)) * m.lz,
                                 std::get<0>(b), id++});
                }
    return a;
}

std::vector<AtomData> gen_cluster(int n, double box)
{
    std::vector<AtomData> a(n);
    std::srand(42);
    double cx = box * 0.25, cy = box * 0.25, cz = box * 0.25, cr = box * 0.06;
    int na = n * 7 / 10;
    for (int i = 0; i < na; i++)
    {
        double r = cr * std::pow(rand() / (double)RAND_MAX, 1.0 / 3.0);
        double th = 6.283185307 * rand() / RAND_MAX;
        double ph = std::acos(2.0 * rand() / RAND_MAX - 1.0);
        a[i] = {cx + r * std::sin(ph) * std::cos(th),
                cy + r * std::sin(ph) * std::sin(th),
                cz + r * std::cos(ph), 0, i};
    }
    for (int i = na; i < n; i++)
        a[i] = {box * rand() / RAND_MAX, box * rand() / RAND_MAX, box * rand() / RAND_MAX, 0, i};
    return a;
}

struct Grid
{
    int nx, ny, nz;
    double x0, y0, z0;
    std::vector<std::vector<std::vector<std::vector<int>>>> g;
};

Grid build_grid(const std::vector<AtomData>& a, double gs)
{
    Grid gr;
    int n = (int)a.size();
    gr.x0 = gr.y0 = gr.z0 = a[0].x;
    double xm = a[0].x, ym = a[0].y, zm = a[0].z;
    for (auto& t : a)
    {
        gr.x0 = std::min(gr.x0, t.x);
        xm = std::max(xm, t.x);
        gr.y0 = std::min(gr.y0, t.y);
        ym = std::max(ym, t.y);
        gr.z0 = std::min(gr.z0, t.z);
        zm = std::max(zm, t.z);
    }
    gr.nx = (int)std::ceil((xm - gr.x0) / gs) + 1;
    gr.ny = (int)std::ceil((ym - gr.y0) / gs) + 1;
    gr.nz = (int)std::ceil((zm - gr.z0) / gs) + 1;
    gr.g.assign(gr.nx, std::vector<std::vector<std::vector<int>>>(
                            gr.ny, std::vector<std::vector<int>>(gr.nz)));
    for (int i = 0; i < n; i++)
    {
        int bx = std::max(0, std::min(gr.nx - 1, (int)std::floor((a[i].x - gr.x0) / gs)));
        int by = std::max(0, std::min(gr.ny - 1, (int)std::floor((a[i].y - gr.y0) / gs)));
        int bz = std::max(0, std::min(gr.nz - 1, (int)std::floor((a[i].z - gr.z0) / gs)));
        gr.g[bx][by][bz].push_back(i);
    }
    return gr;
}

double run_serial(const std::vector<AtomData>& a, double cut, int& nb)
{
    double t0 = omp_get_wtime();
    double gs = cut + 0.1, c2 = cut * cut;
    Grid gr = build_grid(a, gs);
    double cnt = 0;
    int n = (int)a.size(), sr = (int)std::ceil(cut / gs);
    for (int i = 0; i < n; i++)
    {
        int bx = std::max(0, std::min(gr.nx - 1, (int)std::floor((a[i].x - gr.x0) / gs)));
        int by = std::max(0, std::min(gr.ny - 1, (int)std::floor((a[i].y - gr.y0) / gs)));
        int bz = std::max(0, std::min(gr.nz - 1, (int)std::floor((a[i].z - gr.z0) / gs)));
        for (int dx = -sr; dx <= sr; dx++)
        {
            int cx = bx + dx;
            if (cx < 0 || cx >= gr.nx)
                continue;
            for (int dy = -sr; dy <= sr; dy++)
            {
                int cy = by + dy;
                if (cy < 0 || cy >= gr.ny)
                    continue;
                for (int dz = -sr; dz <= sr; dz++)
                {
                    int cz = bz + dz;
                    if (cz < 0 || cz >= gr.nz)
                        continue;
                    for (int j : gr.g[cx][cy][cz])
                    {
                        if (j <= i)
                            continue;
                        double dx2 = a[i].x - a[j].x;
                        double dy2 = a[i].y - a[j].y;
                        double dz2 = a[i].z - a[j].z;
                        if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                            cnt += 1.0;
                    }
                }
            }
        }
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
    int n = (int)a.size();
    double gs = cut + 0.1, c2 = cut * cut;
    Grid gr = build_grid(a, gs);
    int base = n / np, rem = n % np;
    int start = rk * base + std::min(rk, rem);
    int end = start + base + (rk < rem ? 1 : 0);

    double cnt = 0;
#pragma omp parallel
    {
        double lc = 0;
        int sr = (int)std::ceil(cut / gs);
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            int bx = std::max(0, std::min(gr.nx - 1, (int)std::floor((a[i].x - gr.x0) / gs)));
            int by = std::max(0, std::min(gr.ny - 1, (int)std::floor((a[i].y - gr.y0) / gs)));
            int bz = std::max(0, std::min(gr.nz - 1, (int)std::floor((a[i].z - gr.z0) / gs)));
            for (int dx = -sr; dx <= sr; dx++)
            {
                int cx = bx + dx;
                if (cx < 0 || cx >= gr.nx)
                    continue;
                for (int dy = -sr; dy <= sr; dy++)
                {
                    int cy = by + dy;
                    if (cy < 0 || cy >= gr.ny)
                        continue;
                    for (int dz = -sr; dz <= sr; dz++)
                    {
                        int cz = bz + dz;
                        if (cz < 0 || cz >= gr.nz)
                            continue;
                        for (int j : gr.g[cx][cy][cz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = a[i].x - a[j].x;
                            double dy2 = a[i].y - a[j].y;
                            double dz2 = a[i].z - a[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                                lc += 1.0;
                        }
                    }
                }
            }
        }
#pragma omp atomic
        cnt += lc;
    }
    int my = (int)cnt;
    MPI_Allreduce(&my, &nb, 1, MPI_INT, MPI_SUM, comm);
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

    struct M
    {
        std::string n;
        int nx, ny, nz;
        double lx, ly, lz, cut;
        int nt, nat;
        std::vector<std::tuple<int, double, double, double>> bs;
    };
    std::vector<M> mats = {
        {"Al fcc", 10, 5, 5, 2, 2, 2, 1.5, 1, 1000,
         {{0, 0, 0, 0}, {0, 0, 0.5, 0.5}, {0, 0.5, 0, 0.5}, {0, 0.5, 0.5, 0}}},
        {"Si diamond", 10, 5, 5, 2.4, 2.4, 2.4, 1.25, 1, 2000,
         {{0, 0, 0, 0},
          {0, 0, 0.5, 0.5},
          {0, 0.5, 0, 0.5},
          {0, 0.5, 0.5, 0},
          {0, 0.25, 0.25, 0.25},
          {0, 0.25, 0.75, 0.75},
          {0, 0.75, 0.25, 0.75},
          {0, 0.75, 0.75, 0.25}}},
        {"NaCl", 15, 5, 5, 2.4, 2.4, 2.4, 1.35, 2, 3000,
         {{0, 0, 0, 0},
          {0, 0, 0.5, 0.5},
          {0, 0.5, 0, 0.5},
          {0, 0.5, 0.5, 0},
          {1, 0.5, 0, 0},
          {1, 0, 0.5, 0},
          {1, 0, 0, 0.5},
          {1, 0.5, 0.5, 0.5}}},
        {"TiO2 rutile", 10, 10, 7, 2.4, 2.4, 1.56, 1.25, 2, 4200,
         {{0, 0, 0, 0},
          {0, 0.5, 0.5, 0.5},
          {1, 0.305, 0.305, 0},
          {1, 0.695, 0.695, 0},
          {1, 0.805, 0.195, 0.5},
          {1, 0.195, 0.805, 0.5}}},
    };

    if (rk == 0)
    {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  ABACUS Neighbor Search - MPI+OpenMP Benchmark" << std::endl;
        std::cout << "  Cores: " << np * nt << " (" << np << " MPI x " << nt << " OMP)" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }

    struct R
    {
        std::string nm;
        int na;
        double ser, par, sp, eff;
    };
    std::vector<R> rs;

    for (auto& m : mats)
    {
        std::vector<AtomData> at
            = gen_crystal({m.n, m.nx, m.ny, m.nz, m.lx, m.ly, m.lz, m.cut, m.nt, m.nat, m.bs, ""});
        double ser = 0;
        int snb = 0;
        if (rk == 0)
            ser = run_serial(at, m.cut, snb);
        double par = 0;
        int pnb = 0;
        par = run_parallel(at, m.cut, MPI_COMM_WORLD, pnb);
        if (rk == 0)
        {
            double u = np * nt;
            rs.push_back({m.n, (int)at.size(), ser, par, ser / par, ser / par / u * 100});
        }
    }
    {
        double cut = 3.0;
        std::vector<AtomData> at = gen_cluster(5000, 100.0);
        double ser = 0;
        int snb = 0;
        if (rk == 0)
            ser = run_serial(at, cut, snb);
        double par = 0;
        int pnb = 0;
        par = run_parallel(at, cut, MPI_COMM_WORLD, pnb);
        if (rk == 0)
        {
            double u = np * nt;
            rs.push_back({"Cluster", 5000, ser, par, ser / par, ser / par / u * 100});
        }
    }

    if (rk == 0)
    {
        std::cout << "  " << std::left << std::setw(16) << "Case" << std::right << std::setw(7) << "Atoms"
                  << std::setw(12) << "Serial(ms)" << std::setw(12) << "Par(ms)" << std::setw(8) << "Speedup"
                  << std::setw(8) << "Eff%" << std::endl;
        std::cout << "  " << std::string(63, '-') << std::endl;
        double ts = 0, tp = 0;
        for (auto& r : rs)
        {
            ts += r.ser;
            tp += r.par;
            std::cout << "  " << std::left << std::setw(16) << r.nm << std::right << std::setw(7) << r.na
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.ser << std::setw(12)
                      << r.par << std::setw(7) << std::setprecision(1) << r.sp << "x" << std::setw(7)
                      << std::setprecision(1) << r.eff << "%" << std::endl;
        }
        std::cout << "  " << std::string(63, '-') << std::endl;
        double u = np * nt, ov = ts / tp;
        std::cout << "  " << std::left << std::setw(16) << "OVERALL" << std::right << std::setw(7) << ""
                  << std::setw(12) << std::fixed << std::setprecision(2) << ts << std::setw(12) << tp
                  << std::setw(7) << std::setprecision(1) << ov << "x" << std::setw(7)
                  << std::setprecision(1) << (ov / u * 100) << "%" << std::endl;
        std::cout << "============================================================" << std::endl;
    }
    MPI_Finalize();
    return 0;
}
