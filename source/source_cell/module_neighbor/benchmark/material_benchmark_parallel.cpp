#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

std::vector<AtomData> generate_crystal_atoms(const MaterialCase& mat)
{
    std::vector<AtomData> atoms;
    atoms.reserve(mat.expected_atoms);
    int atom_id = 0;
    for (int ix = 0; ix < mat.nx; ++ix)
        for (int iy = 0; iy < mat.ny; ++iy)
            for (int iz = 0; iz < mat.nz; ++iz)
                for (const auto& b : mat.basis)
                {
                    if (atom_id >= mat.expected_atoms)
                        break;
                    AtomData a;
                    a.x = (ix + std::get<1>(b)) * mat.lattice_x;
                    a.y = (iy + std::get<2>(b)) * mat.lattice_y;
                    a.z = (iz + std::get<3>(b)) * mat.lattice_z;
                    a.type = std::get<0>(b);
                    a.id = atom_id++;
                    atoms.push_back(a);
                }
    return atoms;
}

std::vector<AtomData> generate_clustered(int natoms, double box, double frac)
{
    std::vector<AtomData> atoms(natoms);
    int na = static_cast<int>(natoms * frac);
    double cx = box * 0.3, cy = box * 0.3, cz = box * 0.3, cr = box * 0.07;
    std::srand(12345);
    for (int i = 0; i < na; i++)
    {
        double r = cr * std::pow(static_cast<double>(std::rand()) / RAND_MAX, 1.0 / 3.0);
        double th = 6.283185307 * std::rand() / RAND_MAX;
        double ph = std::acos(2.0 * std::rand() / RAND_MAX - 1.0);
        atoms[i].x = cx + r * std::sin(ph) * std::cos(th);
        atoms[i].y = cy + r * std::sin(ph) * std::sin(th);
        atoms[i].z = cz + r * std::cos(ph);
        atoms[i].type = 0;
        atoms[i].id = i;
    }
    for (int i = na; i < natoms; i++)
    {
        atoms[i].x = box * std::rand() / RAND_MAX;
        atoms[i].y = box * std::rand() / RAND_MAX;
        atoms[i].z = box * std::rand() / RAND_MAX;
        atoms[i].type = 0;
        atoms[i].id = i;
    }
    return atoms;
}

struct GridInfo
{
    int nx, ny, nz;
    double x_min, y_min, z_min;
    std::vector<std::vector<std::vector<std::vector<int>>>> grid;
};

GridInfo build_grid_local(const std::vector<AtomData>& atoms, double grid_size)
{
    GridInfo g;
    int n = static_cast<int>(atoms.size());
    g.x_min = g.y_min = g.z_min = atoms[0].x;
    double xm = atoms[0].x, ym = atoms[0].y, zm = atoms[0].z;
    for (const auto& a : atoms)
    {
        g.x_min = std::min(g.x_min, a.x);
        xm = std::max(xm, a.x);
        g.y_min = std::min(g.y_min, a.y);
        ym = std::max(ym, a.y);
        g.z_min = std::min(g.z_min, a.z);
        zm = std::max(zm, a.z);
    }
    g.nx = static_cast<int>(std::ceil((xm - g.x_min) / grid_size)) + 1;
    g.ny = static_cast<int>(std::ceil((ym - g.y_min) / grid_size)) + 1;
    g.nz = static_cast<int>(std::ceil((zm - g.z_min) / grid_size)) + 1;

    g.grid.assign(g.nx, std::vector<std::vector<std::vector<int>>>(g.ny, std::vector<std::vector<int>>(g.nz)));
    for (int i = 0; i < n; i++)
    {
        int bx = std::max(0, std::min(g.nx - 1, (int)std::floor((atoms[i].x - g.x_min) / grid_size)));
        int by = std::max(0, std::min(g.ny - 1, (int)std::floor((atoms[i].y - g.y_min) / grid_size)));
        int bz = std::max(0, std::min(g.nz - 1, (int)std::floor((atoms[i].z - g.z_min) / grid_size)));
        g.grid[bx][by][bz].push_back(i);
    }
    return g;
}

std::vector<double> compute_weights(const std::vector<AtomData>& atoms,
                                   double cutoff,
                                   double grid_size,
                                   const GridInfo& g)
{
    int n = (int)atoms.size();
    std::vector<double> w(n, 1.0);
    int sr = (int)std::ceil(cutoff / grid_size);
    for (int i = 0; i < n; i++)
    {
        int bxi = std::max(0, std::min(g.nx - 1, (int)std::floor((atoms[i].x - g.x_min) / grid_size)));
        int byi = std::max(0, std::min(g.ny - 1, (int)std::floor((atoms[i].y - g.y_min) / grid_size)));
        int bzi = std::max(0, std::min(g.nz - 1, (int)std::floor((atoms[i].z - g.z_min) / grid_size)));
        int nb = 0;
        for (int dx = -sr; dx <= sr; dx++)
        {
            int bx = bxi + dx;
            if (bx < 0 || bx >= g.nx)
                continue;
            for (int dy = -sr; dy <= sr; dy++)
            {
                int by = byi + dy;
                if (by < 0 || by >= g.ny)
                    continue;
                for (int dz = -sr; dz <= sr; dz++)
                {
                    int bz = bzi + dz;
                    if (bz < 0 || bz >= g.nz)
                        continue;
                    nb += (int)g.grid[bx][by][bz].size();
                }
            }
        }
        w[i] = std::max(1.0, (double)nb);
    }
    return w;
}

struct Decomp
{
    std::vector<int> offsets;
    double imb;
};

Decomp distribute_weighted(int n, const std::vector<double>& w, int np)
{
    std::vector<double> pref(n + 1, 0.0);
    for (int i = 0; i < n; i++)
        pref[i + 1] = pref[i] + w[i];
    double tot = pref[n];
    double per = tot / np;
    Decomp d;
    d.offsets.resize(np + 1, 0);
    d.offsets[np] = n;
    int cur = 0;
    for (int p = 1; p < np; p++)
    {
        double tgt = p * per;
        while (cur < n && pref[cur + 1] < tgt)
            cur++;
        d.offsets[p] = std::min(n, cur + 1);
    }
    double mx = 0, mn = 1e100;
    for (int p = 0; p < np; p++)
    {
        double load = pref[d.offsets[p + 1]] - pref[d.offsets[p]];
        mx = std::max(mx, load);
        mn = std::min(mn, load);
    }
    d.imb = (mn > 0) ? mx / mn : 1.0;
    return d;
}

double serial_search(const std::vector<AtomData>& atoms, double cutoff, int& total_nb)
{
    double t0 = omp_get_wtime();
    int n = (int)atoms.size();
    double c2 = cutoff * cutoff;
    double gs = cutoff + 0.1;
    GridInfo g = build_grid_local(atoms, gs);
    total_nb = 0;

    for (int i = 0; i < n; i++)
    {
        int bxi = std::max(0, std::min(g.nx - 1, (int)std::floor((atoms[i].x - g.x_min) / gs)));
        int byi = std::max(0, std::min(g.ny - 1, (int)std::floor((atoms[i].y - g.y_min) / gs)));
        int bzi = std::max(0, std::min(g.nz - 1, (int)std::floor((atoms[i].z - g.z_min) / gs)));
        int sr = (int)std::ceil(cutoff / gs);
        for (int dx = -sr; dx <= sr; dx++)
        {
            int bx = bxi + dx;
            if (bx < 0 || bx >= g.nx)
                continue;
            for (int dy = -sr; dy <= sr; dy++)
            {
                int by = byi + dy;
                if (by < 0 || by >= g.ny)
                    continue;
                for (int dz = -sr; dz <= sr; dz++)
                {
                    int bz = bzi + dz;
                    if (bz < 0 || bz >= g.nz)
                        continue;
                    for (int j : g.grid[bx][by][bz])
                    {
                        if (j <= i)
                            continue;
                        double dx2 = atoms[i].x - atoms[j].x;
                        double dy2 = atoms[i].y - atoms[j].y;
                        double dz2 = atoms[i].z - atoms[j].z;
                        if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                            total_nb++;
                    }
                }
            }
        }
    }
    return (omp_get_wtime() - t0) * 1000.0;
}

double par_static(const std::vector<AtomData>& atoms,
                  double cutoff,
                  MPI_Comm comm,
                  int& total_nb,
                  double& imb,
                  double& t_comp_max,
                  double& t_comp_min)
{
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    double t0 = MPI_Wtime();
    int n = (int)atoms.size();
    double c2 = cutoff * cutoff;
    double gs = cutoff + 0.1;

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    std::vector<double> buf(n * 3);
    if (rank == 0)
        for (int i = 0; i < n; i++)
        {
            buf[i * 3] = atoms[i].x;
            buf[i * 3 + 1] = atoms[i].y;
            buf[i * 3 + 2] = atoms[i].z;
        }
    MPI_Bcast(buf.data(), n * 3, MPI_DOUBLE, 0, comm);

    std::vector<AtomData> la(n);
    for (int i = 0; i < n; i++)
    {
        la[i].x = buf[i * 3];
        la[i].y = buf[i * 3 + 1];
        la[i].z = buf[i * 3 + 2];
        la[i].id = i;
    }
    GridInfo g = build_grid_local(la, gs);

    int base = n / np, rem = n % np;
    int start = rank * base + std::min(rank, rem);
    int end = start + base + (rank < rem ? 1 : 0);
    int cnt = end - start;

    std::vector<int> all_cnt(np);
    MPI_Allgather(&cnt, 1, MPI_INT, all_cnt.data(), 1, MPI_INT, comm);
    double imb_static = 1.0;
    {
        int mx = *std::max_element(all_cnt.begin(), all_cnt.end());
        int mn = *std::min_element(all_cnt.begin(), all_cnt.end());
        imb_static = (mn > 0) ? (double)mx / mn : 1.0;
    }

    double t_comp_start = MPI_Wtime();
    double my_wk = 0;

#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            int bxi = std::max(0, std::min(g.nx - 1, (int)std::floor((la[i].x - g.x_min) / gs)));
            int byi = std::max(0, std::min(g.ny - 1, (int)std::floor((la[i].y - g.y_min) / gs)));
            int bzi = std::max(0, std::min(g.nz - 1, (int)std::floor((la[i].z - g.z_min) / gs)));
            int sr = (int)std::ceil(cutoff / gs);
            for (int dx = -sr; dx <= sr; dx++)
            {
                int bx = bxi + dx;
                if (bx < 0 || bx >= g.nx)
                    continue;
                for (int dy = -sr; dy <= sr; dy++)
                {
                    int by = byi + dy;
                    if (by < 0 || by >= g.ny)
                        continue;
                    for (int dz = -sr; dz <= sr; dz++)
                    {
                        int bz = bzi + dz;
                        if (bz < 0 || bz >= g.nz)
                            continue;
                        for (int j : g.grid[bx][by][bz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = la[i].x - la[j].x;
                            double dy2 = la[i].y - la[j].y;
                            double dz2 = la[i].z - la[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                                my_wk += 1.0;
                        }
                    }
                }
            }
        }
    }
    double t_comp_end = MPI_Wtime();
    double t_comp = t_comp_end - t_comp_start;

    std::vector<double> all_t(np);
    MPI_Gather(&t_comp, 1, MPI_DOUBLE, all_t.data(), 1, MPI_DOUBLE, 0, comm);

    int my_nb = (int)my_wk;
    MPI_Reduce(&my_nb, &total_nb, 1, MPI_INT, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        imb = imb_static;
        t_comp_max = *std::max_element(all_t.begin(), all_t.end());
        t_comp_min = *std::min_element(all_t.begin(), all_t.end());
    }

    return (MPI_Wtime() - t0) * 1000.0;
}

double par_dynamic(const std::vector<AtomData>& atoms,
                   double cutoff,
                   MPI_Comm comm,
                   int& total_nb,
                   double& imb,
                   double& t_comp_max,
                   double& t_comp_min)
{
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    double t0 = MPI_Wtime();
    int n = (int)atoms.size();
    double c2 = cutoff * cutoff;
    double gs = cutoff + 0.1;

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    std::vector<double> buf(n * 3);
    if (rank == 0)
        for (int i = 0; i < n; i++)
        {
            buf[i * 3] = atoms[i].x;
            buf[i * 3 + 1] = atoms[i].y;
            buf[i * 3 + 2] = atoms[i].z;
        }
    MPI_Bcast(buf.data(), n * 3, MPI_DOUBLE, 0, comm);

    std::vector<AtomData> la(n);
    for (int i = 0; i < n; i++)
    {
        la[i].x = buf[i * 3];
        la[i].y = buf[i * 3 + 1];
        la[i].z = buf[i * 3 + 2];
        la[i].id = i;
    }
    GridInfo g = build_grid_local(la, gs);

    std::vector<double> w;
    Decomp decomp;
    decomp.offsets.resize(np + 1);

    if (rank == 0)
    {
        w = compute_weights(la, cutoff, gs, g);
        decomp = distribute_weighted(n, w, np);
    }

    MPI_Bcast(decomp.offsets.data(), np + 1, MPI_INT, 0, comm);
    MPI_Bcast(&decomp.imb, 1, MPI_DOUBLE, 0, comm);

    int start = decomp.offsets[rank];
    int end = decomp.offsets[rank + 1];

    double t_comp_start = MPI_Wtime();
    double my_wk = 0;

#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            int bxi = std::max(0, std::min(g.nx - 1, (int)std::floor((la[i].x - g.x_min) / gs)));
            int byi = std::max(0, std::min(g.ny - 1, (int)std::floor((la[i].y - g.y_min) / gs)));
            int bzi = std::max(0, std::min(g.nz - 1, (int)std::floor((la[i].z - g.z_min) / gs)));
            int sr = (int)std::ceil(cutoff / gs);
            for (int dx = -sr; dx <= sr; dx++)
            {
                int bx = bxi + dx;
                if (bx < 0 || bx >= g.nx)
                    continue;
                for (int dy = -sr; dy <= sr; dy++)
                {
                    int by = byi + dy;
                    if (by < 0 || by >= g.ny)
                        continue;
                    for (int dz = -sr; dz <= sr; dz++)
                    {
                        int bz = bzi + dz;
                        if (bz < 0 || bz >= g.nz)
                            continue;
                        for (int j : g.grid[bx][by][bz])
                        {
                            if (j <= i)
                                continue;
                            double dx2 = la[i].x - la[j].x;
                            double dy2 = la[i].y - la[j].y;
                            double dz2 = la[i].z - la[j].z;
                            if (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < c2)
                                my_wk += 1.0;
                        }
                    }
                }
            }
        }
    }
    double t_comp_end = MPI_Wtime();
    double t_comp = t_comp_end - t_comp_start;

    std::vector<double> all_t(np);
    MPI_Gather(&t_comp, 1, MPI_DOUBLE, all_t.data(), 1, MPI_DOUBLE, 0, comm);

    int my_nb = (int)my_wk;
    MPI_Reduce(&my_nb, &total_nb, 1, MPI_INT, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        imb = decomp.imb;
        t_comp_max = *std::max_element(all_t.begin(), all_t.end());
        t_comp_min = *std::min_element(all_t.begin(), all_t.end());
    }

    return (MPI_Wtime() - t0) * 1000.0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int nt = 1;
#pragma omp parallel
    {
#pragma omp master
        nt = omp_get_num_threads();
    }

    const MaterialCase mats[] = {
        {"Al fcc", 10, 5, 5, 2.0, 2.0, 2.0, 1.5, 1,
         {{0, 0., 0., 0.}, {0, 0., 0.5, 0.5}, {0, 0.5, 0., 0.5}, {0, 0.5, 0.5, 0.}}, 1000, "uniform crystal"},
        {"Si diamond", 10, 5, 5, 2.4, 2.4, 2.4, 1.25, 1,
         {{0, 0., 0., 0.},
          {0, 0., 0.5, 0.5},
          {0, 0.5, 0., 0.5},
          {0, 0.5, 0.5, 0.},
          {0, 0.25, 0.25, 0.25},
          {0, 0.25, 0.75, 0.75},
          {0, 0.75, 0.25, 0.75},
          {0, 0.75, 0.75, 0.25}},
         2000, "uniform crystal"},
        {"NaCl", 15, 5, 5, 2.4, 2.4, 2.4, 1.35, 2,
         {{0, 0., 0., 0.},
          {0, 0., 0.5, 0.5},
          {0, 0.5, 0., 0.5},
          {0, 0.5, 0.5, 0.},
          {1, 0.5, 0., 0.},
          {1, 0., 0.5, 0.},
          {1, 0., 0., 0.5},
          {1, 0.5, 0.5, 0.5}},
         3000, "uniform crystal"},
        {"TiO2 rutile", 10, 10, 7, 2.4, 2.4, 1.56, 1.25, 2,
         {{0, 0., 0., 0.},
          {0, 0.5, 0.5, 0.5},
          {1, 0.305, 0.305, 0.},
          {1, 0.695, 0.695, 0.},
          {1, 0.805, 0.195, 0.5},
          {1, 0.195, 0.805, 0.5}},
         4200, "uniform crystal"},
    };

    if (rank == 0)
    {
        std::cout << "\n======================================================================" << std::endl;
        std::cout << "  ABACUS Neighbor Search - Dynamic Load Balancing Benchmark" << std::endl;
        std::cout << "  MPI x OMP: " << np << " x " << nt << " = " << np * nt << " units" << std::endl;
        std::cout << "======================================================================" << std::endl;
    }

    struct Row
    {
        std::string name;
        int natoms;
        double serial, stime, ssu, seff, simb, stmax, stmin, dtime, dsu, deff, dimb, dtmax, dtmin;
    };
    std::vector<Row> rows;

    for (int mi = 0; mi < 4; mi++)
    {
        const auto& mat = mats[mi];
        std::vector<AtomData> atoms;
        if (rank == 0)
            atoms = generate_crystal_atoms(mat);

        double ser = 0;
        int snb = 0;
        if (rank == 0)
            ser = serial_search(atoms, mat.cutoff, snb);

        MPI_Barrier(MPI_COMM_WORLD);

        double st = 0, simb = 1, stmax = 0, stmin = 0;
        int stnb = 0;
        st = par_static(atoms, mat.cutoff, MPI_COMM_WORLD, stnb, simb, stmax, stmin);

        MPI_Barrier(MPI_COMM_WORLD);

        double dt = 0, dimb = 1, dtmax = 0, dtmin = 0;
        int dtnb = 0;
        dt = par_dynamic(atoms, mat.cutoff, MPI_COMM_WORLD, dtnb, dimb, dtmax, dtmin);

        if (rank == 0)
        {
            double us = np * nt;
            rows.push_back({mat.name, mat.expected_atoms, ser, st, (ser / st), (ser / st / us * 100),
                            simb, stmax * 1000, stmin * 1000, dt, (ser / dt), (ser / dt / us * 100), dimb,
                            dtmax * 1000, dtmin * 1000});
        }
    }

    {
        double cutoff = 3.0;
        std::vector<AtomData> atoms;
        if (rank == 0)
            atoms = generate_clustered(5000, 100.0, 0.7);

        double ser = 0;
        int snb = 0;
        if (rank == 0)
            ser = serial_search(atoms, cutoff, snb);

        MPI_Barrier(MPI_COMM_WORLD);

        double st = 0, simb = 1, stmax = 0, stmin = 0;
        int stnb = 0;
        st = par_static(atoms, cutoff, MPI_COMM_WORLD, stnb, simb, stmax, stmin);

        MPI_Barrier(MPI_COMM_WORLD);

        double dt = 0, dimb = 1, dtmax = 0, dtmin = 0;
        int dtnb = 0;
        dt = par_dynamic(atoms, cutoff, MPI_COMM_WORLD, dtnb, dimb, dtmax, dtmin);

        if (rank == 0)
        {
            double us = np * nt;
            rows.push_back({"Cluster 70%", 5000, ser, st, (ser / st), (ser / st / us * 100), simb,
                            stmax * 1000, stmin * 1000, dt, (ser / dt), (ser / dt / us * 100), dimb,
                            dtmax * 1000, dtmin * 1000});
        }
    }

    if (rank == 0)
    {
        std::cout << "\n  " << std::left << std::setw(16) << "Case"
                  << std::right << std::setw(6) << "Atoms"
                  << std::setw(10) << "Serial"
                  << std::setw(10) << "Static"
                  << std::setw(6) << "SU"
                  << std::setw(7) << "Eff%"
                  << std::setw(8) << "ImbS"
                  << std::setw(8) << "TmaxS"
                  << std::setw(8) << "TminS"
                  << std::setw(10) << "Dynamic"
                  << std::setw(6) << "SU"
                  << std::setw(7) << "Eff%"
                  << std::setw(8) << "ImbD"
                  << std::setw(8) << "TmaxD"
                  << std::setw(8) << "TminD" << std::endl;
        std::cout << "  " << std::string(125, '-') << std::endl;

        double ts = 0, tsta = 0, tdyn = 0;
        for (const auto& r : rows)
        {
            ts += r.serial;
            tsta += r.stime;
            tdyn += r.dtime;
            std::cout << "  " << std::left << std::setw(16) << r.name << std::right << std::setw(6)
                      << r.natoms << std::setw(10) << std::fixed << std::setprecision(1) << r.serial
                      << std::setw(10) << std::fixed << std::setprecision(1) << r.stime << std::setw(5)
                      << std::fixed << std::setprecision(1) << r.ssu << "x" << std::setw(6) << std::fixed
                      << std::setprecision(1) << r.seff << "%" << std::setw(8) << std::fixed
                      << std::setprecision(2) << r.simb << std::setw(8) << std::fixed << std::setprecision(1)
                      << r.stmax << std::setw(8) << std::fixed << std::setprecision(1) << r.stmin
                      << std::setw(10) << std::fixed << std::setprecision(1) << r.dtime << std::setw(5)
                      << std::fixed << std::setprecision(1) << r.dsu << "x" << std::setw(6) << std::fixed
                      << std::setprecision(1) << r.deff << "%" << std::setw(8) << std::fixed
                      << std::setprecision(2) << r.dimb << std::setw(8) << std::fixed << std::setprecision(1)
                      << r.dtmax << std::setw(8) << std::fixed << std::setprecision(1) << r.dtmin << std::endl;
        }
        std::cout << "  " << std::string(125, '-') << std::endl;

        double us = np * nt;
        double os = ts / tsta, od = ts / tdyn;
        std::cout << "  " << std::left << std::setw(16) << "OVERALL" << std::right << std::setw(6) << ""
                  << std::setw(10) << std::fixed << std::setprecision(1) << ts << std::setw(10) << std::fixed
                  << std::setprecision(1) << tsta << std::setw(5) << std::fixed << std::setprecision(1) << os
                  << "x" << std::setw(6) << std::fixed << std::setprecision(1) << (os / us * 100) << "%"
                  << std::setw(8) << "-" << std::setw(8) << "-" << std::setw(8) << "-" << std::setw(10)
                  << std::fixed << std::setprecision(1) << tdyn << std::setw(5) << std::fixed
                  << std::setprecision(1) << od << "x" << std::setw(6) << std::fixed << std::setprecision(1)
                  << (od / us * 100) << "%" << std::setw(8) << "-" << std::setw(8) << "-" << std::setw(8)
                  << "-" << std::endl;

        std::cout << "\n  ImbS/D = atom-count imbalance (1.0 = perfect)"
                  << "\n  Tmax/min = per-process compute time (ms), smaller gap = better balance"
                  << "\n======================================================================" << std::endl;
        std::cout << "  Static:  " << std::fixed << std::setprecision(2) << os << "x, " << std::setprecision(1)
                  << (os / us * 100) << "% eff" << std::endl;
        std::cout << "  Dynamic: " << std::fixed << std::setprecision(2) << od << "x, " << std::setprecision(1)
                  << (od / us * 100) << "% eff" << std::endl;
        std::cout << "======================================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
