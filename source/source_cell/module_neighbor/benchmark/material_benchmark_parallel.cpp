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
    std::string name; int nx, ny, nz;
    double lx, ly, lz, cutoff; int ntype, expected_atoms;
    std::vector<std::tuple<int, double, double, double>> basis;
};

std::vector<AtomData> gen_crystal(const Mate& m)
{
    std::vector<AtomData> a; a.reserve(m.expected_atoms);
    int id = 0;
    for (int ix = 0; ix < m.nx; ++ix)
        for (int iy = 0; iy < m.ny; ++iy)
            for (int iz = 0; iz < m.nz; ++iz)
                for (auto& b : m.basis) {
                    if (id >= m.expected_atoms) break;
                    a.push_back({(ix + std::get<1>(b)) * m.lx,
                                 (iy + std::get<2>(b)) * m.ly,
                                 (iz + std::get<3>(b)) * m.lz,
                                 std::get<0>(b), id++});
                }
    return a;
}

struct Grid
{
    int nx, ny, nz;
    double x0, y0, z0;
    std::vector<int> cell_offsets;
    std::vector<int> cell_data;
    std::vector<int> non_empty;
};

Grid build_grid(const std::vector<AtomData>& a, double gs)
{
    Grid gr; int n = (int)a.size();
    gr.x0 = gr.y0 = gr.z0 = a[0].x;
    double xm = a[0].x, ym = a[0].y, zm = a[0].z;
    for (auto& t : a) {
        gr.x0 = std::min(gr.x0, t.x); xm = std::max(xm, t.x);
        gr.y0 = std::min(gr.y0, t.y); ym = std::max(ym, t.y);
        gr.z0 = std::min(gr.z0, t.z); zm = std::max(zm, t.z);
    }
    gr.nx = (int)std::ceil((xm - gr.x0) / gs) + 1;
    gr.ny = (int)std::ceil((ym - gr.y0) / gs) + 1;
    gr.nz = (int)std::ceil((zm - gr.z0) / gs) + 1;

    int total = gr.nx * gr.ny * gr.nz;
    std::vector<std::vector<int>> bins(total);
    for (int i = 0; i < n; i++) {
        int bx = std::max(0, std::min(gr.nx - 1, (int)std::floor((a[i].x - gr.x0) / gs)));
        int by = std::max(0, std::min(gr.ny - 1, (int)std::floor((a[i].y - gr.y0) / gs)));
        int bz = std::max(0, std::min(gr.nz - 1, (int)std::floor((a[i].z - gr.z0) / gs)));
        bins[bx * gr.ny * gr.nz + by * gr.nz + bz].push_back(i);
    }

    gr.cell_offsets.resize(total + 1, 0);
    for (int i = 0; i < total; i++) {
        gr.cell_offsets[i + 1] = gr.cell_offsets[i] + (int)bins[i].size();
        if (!bins[i].empty()) gr.non_empty.push_back(i);
    }
    gr.cell_data.resize(gr.cell_offsets[total]);
    for (int i = 0; i < total; i++)
        std::copy(bins[i].begin(), bins[i].end(), gr.cell_data.begin() + gr.cell_offsets[i]);

    return gr;
}

double run_serial(const std::vector<AtomData>& a, double cut, int& nb)
{
    double t0 = omp_get_wtime(), gs = cut + 0.1, c2 = cut * cut, cnt = 0;
    int n = (int)a.size();
    Grid gr = build_grid(a, gs);

    for (int i = 0; i < n; i++) {
        for (int ci : gr.non_empty) {
            int bz = ci % gr.nz, tmp = ci / gr.nz;
            int by = tmp % gr.ny, bx = tmp / gr.ny;
            int off = gr.cell_offsets[ci], count = gr.cell_offsets[ci + 1] - off;
            auto it = std::lower_bound(gr.cell_data.begin() + off, gr.cell_data.begin() + off + count, i + 1);
            for (; it != gr.cell_data.begin() + off + count; ++it) {
                int j = *it;
                double dx = a[i].x - a[j].x;
                double dy = a[i].y - a[j].y;
                double dz = a[i].z - a[j].z;
                if (dx * dx + dy * dy + dz * dz < c2) cnt += 1.0;
            }
        }
    }
    nb = (int)cnt;
    return (omp_get_wtime() - t0) * 1000.0;
}

double run_parallel(const std::vector<AtomData>& a, double cut, MPI_Comm comm, int& nb)
{
    int rk, np; MPI_Comm_rank(comm, &rk); MPI_Comm_size(comm, &np);
    double t0 = MPI_Wtime(), gs = cut + 0.1, c2 = cut * cut;
    int n = (int)a.size();
    Grid gr = build_grid(a, gs);
    int nne = (int)gr.non_empty.size();

    int base = n / np, rem = n % np;
    int start = rk * base + std::min(rk, rem);
    int end = start + base + (rk < rem ? 1 : 0);

    double cnt = 0;
#pragma omp parallel
    {
        double lc = 0;
#pragma omp for schedule(guided) nowait
        for (int i = start; i < end; i++) {
            for (int ci = 0; ci < nne; ci++) {
                int cidx = gr.non_empty[ci];
                int off = gr.cell_offsets[cidx], count = gr.cell_offsets[cidx + 1] - off;
                auto it = std::lower_bound(gr.cell_data.begin() + off, gr.cell_data.begin() + off + count, i + 1);
                for (; it != gr.cell_data.begin() + off + count; ++it) {
                    int j = *it;
                    double dx = a[i].x - a[j].x;
                    double dy = a[i].y - a[j].y;
                    double dz = a[i].z - a[j].z;
                    if (dx * dx + dy * dy + dz * dz < c2) lc += 1.0;
                }
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
    int rk, np; MPI_Comm_rank(MPI_COMM_WORLD, &rk); MPI_Comm_size(MPI_COMM_WORLD, &np);
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

    if (rk == 0) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  ABACUS Neighbor Search - MPI+OpenMP Benchmark" << std::endl;
        std::cout << "  Cores: " << np * nt << " (" << np << " MPI x " << nt << " OMP)" << std::endl;
        std::cout << "  Opt: non-empty cells + binary-search j>i + guided schedule" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }

    struct R { std::string nm; int na; double ser, par, sp, eff; };
    std::vector<R> rs;

    for (auto& m : mats) {
        std::vector<AtomData> at = gen_crystal(m);
        double ser = 0; int snb = 0; if (rk == 0) ser = run_serial(at, m.cutoff, snb);
        MPI_Barrier(MPI_COMM_WORLD);
        double par = 0; int pnb = 0; par = run_parallel(at, m.cutoff, MPI_COMM_WORLD, pnb);
        if (rk == 0) {
            double u = np * nt;
            rs.push_back({m.name, (int)at.size(), ser, par, ser / par, ser / par / u * 100.0});
        }
    }

    if (rk == 0) {
        std::cout << "  " << std::left << std::setw(14) << "Case" << std::right << std::setw(8) << "Atoms"
                  << std::setw(10) << "Ser(ms)" << std::setw(10) << "Par(ms)" << std::setw(8) << "Sp"
                  << std::setw(7) << "Eff" << std::endl;
        std::cout << "  " << std::string(57, '-') << std::endl;
        double ts = 0, tp = 0;
        for (auto& r : rs) {
            ts += r.ser; tp += r.par;
            std::cout << "  " << std::left << std::setw(14) << r.nm << std::right << std::setw(8) << r.na
                      << std::setw(10) << std::fixed << std::setprecision(0) << r.ser
                      << std::setw(10) << r.par << std::setw(7) << std::setprecision(1) << r.sp << "x"
                      << std::setw(6) << std::setprecision(1) << r.eff << "%" << std::endl;
        }
        std::cout << "  " << std::string(57, '-') << std::endl;
        double u = np * nt, ov = ts / tp;
        std::cout << "  " << std::left << std::setw(14) << "OVERALL" << std::right << std::setw(8) << ""
                  << std::setw(10) << std::fixed << std::setprecision(0) << ts << std::setw(10) << tp
                  << std::setw(7) << std::setprecision(1) << ov << "x" << std::setw(6)
                  << std::setprecision(1) << (ov / u * 100.0) << "%" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }
    MPI_Finalize(); return 0;
}
