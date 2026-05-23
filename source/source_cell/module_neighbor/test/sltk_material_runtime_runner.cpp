#include "../sltk_grid_driver.h"

#include "source_base/module_external/blas_connector.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Minimal definitions for linking the standalone runner without the full ABACUS executable.
pseudo::pseudo() {}
pseudo::~pseudo() {}

Atom_pseudo::Atom_pseudo() {}
Atom_pseudo::~Atom_pseudo() {}

Atom::Atom() {}
Atom::~Atom() {}

Sep_Cell::Sep_Cell() noexcept {}
Sep_Cell::~Sep_Cell() noexcept {}

namespace ModuleBase
{
namespace GlobalFunc
{
size_t MemAvailable()
{
    return 0;
}
} // namespace GlobalFunc
} // namespace ModuleBase

Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}
Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}

UnitCell::UnitCell() {}
UnitCell::~UnitCell()
{
    if (this->set_atom_flag)
    {
        delete[] this->atoms;
    }
}

void BlasConnector::gemm(const char,
                         const char,
                         const int,
                         const int,
                         const int,
                         const double,
                         const double*,
                         const int,
                         const double*,
                         const int,
                         const double,
                         double*,
                         const int,
                         base_device::AbacusDevice_t)
{
}

double BlasConnector::nrm2(const int n, const double* x, const int incx, base_device::AbacusDevice_t)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum += x[i * incx] * x[i * incx];
    }
    return std::sqrt(sum);
}

void BlasConnector::gemm(const char,
                         const char,
                         const int,
                         const int,
                         const int,
                         const std::complex<double>,
                         const std::complex<double>*,
                         const int,
                         const std::complex<double>*,
                         const int,
                         const std::complex<double>,
                         std::complex<double>*,
                         const int,
                         base_device::AbacusDevice_t)
{
}

namespace
{
struct MaterialCase
{
    std::string name;
    int nx;
    int ny;
    int nz;
    double lattice_x;
    double lattice_y;
    double lattice_z;
    double cutoff;
    int ntype;
    std::vector<std::tuple<int, double, double, double>> basis;
    int expected_atoms;
    double expected_average_neighbors;
};

std::unique_ptr<UnitCell> make_crystal_case(const MaterialCase& material)
{
    std::unique_ptr<UnitCell> ucell(new UnitCell);
    ucell->lat0 = 1.0;
    ucell->latvec.e11 = material.nx * material.lattice_x;
    ucell->latvec.e12 = 0.0;
    ucell->latvec.e13 = 0.0;
    ucell->latvec.e21 = 0.0;
    ucell->latvec.e22 = material.ny * material.lattice_y;
    ucell->latvec.e23 = 0.0;
    ucell->latvec.e31 = 0.0;
    ucell->latvec.e32 = 0.0;
    ucell->latvec.e33 = material.nz * material.lattice_z;
    ucell->a1 = ModuleBase::Vector3<double>(ucell->latvec.e11, ucell->latvec.e12, ucell->latvec.e13);
    ucell->a2 = ModuleBase::Vector3<double>(ucell->latvec.e21, ucell->latvec.e22, ucell->latvec.e23);
    ucell->a3 = ModuleBase::Vector3<double>(ucell->latvec.e31, ucell->latvec.e32, ucell->latvec.e33);
    ucell->omega = ucell->latvec.e11 * ucell->latvec.e22 * ucell->latvec.e33;
    ucell->ntype = material.ntype;
    ucell->nat = material.expected_atoms;
    ucell->atoms = new Atom[material.ntype];
    ucell->set_atom_flag = true;

    std::vector<std::vector<ModuleBase::Vector3<double>>> atoms_by_type(material.ntype);
    int generated_atoms = 0;
    for (int ix = 0; ix < material.nx; ++ix)
    {
        for (int iy = 0; iy < material.ny; ++iy)
        {
            for (int iz = 0; iz < material.nz; ++iz)
            {
                for (const auto& atom_basis : material.basis)
                {
                    if (generated_atoms >= material.expected_atoms)
                    {
                        break;
                    }
                    const int type = std::get<0>(atom_basis);
                    const double fx = std::get<1>(atom_basis);
                    const double fy = std::get<2>(atom_basis);
                    const double fz = std::get<3>(atom_basis);
                    atoms_by_type[type].push_back(ModuleBase::Vector3<double>(
                        (ix + fx) * material.lattice_x,
                        (iy + fy) * material.lattice_y,
                        (iz + fz) * material.lattice_z));
                    ++generated_atoms;
                }
            }
        }
    }

    for (int type = 0; type < material.ntype; ++type)
    {
        ucell->atoms[type].type = type;
        ucell->atoms[type].label = "T" + std::to_string(type);
        ucell->atoms[type].na = static_cast<int>(atoms_by_type[type].size());
        ucell->atoms[type].tau = atoms_by_type[type];
    }

    return ucell;
}

void run_case(const MaterialCase& material, std::ofstream& ofs)
{
    std::unique_ptr<UnitCell> ucell = make_crystal_case(material);
    if (ucell->nat != material.expected_atoms)
    {
        throw std::runtime_error(material.name + ": atom count mismatch");
    }

    Grid_Driver grid_d;
    const auto total_start = std::chrono::steady_clock::now();
    const auto build_start = std::chrono::steady_clock::now();
    grid_d.init(ofs, *ucell, material.cutoff, true);
    const auto build_finish = std::chrono::steady_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(build_finish - build_start).count();

    long long total_neighbors = 0;
    int atoms_with_neighbors = 0;
    AdjacentAtomInfo adjs;
    const auto traverse_start = std::chrono::steady_clock::now();
    for (int type = 0; type < ucell->ntype; ++type)
    {
        for (int atom = 0; atom < ucell->atoms[type].na; ++atom)
        {
            grid_d.Find_atom(*ucell, type, atom, &adjs);
            total_neighbors += adjs.adj_num;
            if (adjs.adj_num > 0)
            {
                ++atoms_with_neighbors;
            }
            if (adjs.ntype.empty() || adjs.ntype.back() != type || adjs.natom.back() != atom)
            {
                throw std::runtime_error(material.name + ": self atom is not the final adjacent entry");
            }
        }
    }
    const auto traverse_finish = std::chrono::steady_clock::now();
    const double traverse_ms = std::chrono::duration<double, std::milli>(traverse_finish - traverse_start).count();
    const double total_ms = std::chrono::duration<double, std::milli>(traverse_finish - total_start).count();

    if (atoms_with_neighbors != material.expected_atoms || total_neighbors <= material.expected_atoms)
    {
        throw std::runtime_error(material.name + ": neighbor coverage is too weak");
    }

    const double average_neighbors = static_cast<double>(total_neighbors) / material.expected_atoms;
    if (material.expected_average_neighbors > 0.0
        && std::abs(average_neighbors - material.expected_average_neighbors) > 1.0e-12)
    {
        throw std::runtime_error(material.name + ": average neighbor count mismatch");
    }
    std::cout << "[sltk] " << material.name << ": atoms=" << material.expected_atoms
              << ", avg_neighbors=" << average_neighbors << ", build_ms=" << build_ms
              << ", traverse_ms=" << traverse_ms << ", total_ms=" << total_ms << std::endl;
}
} // namespace

int main()
{
    const std::vector<MaterialCase> materials = {
        {"Al fcc / 1000 atoms / PW / metal",
         10,
         5,
         5,
         2.0,
         2.0,
         2.0,
         1.5,
         1,
         {{0, 0.0, 0.0, 0.0}, {0, 0.0, 0.5, 0.5}, {0, 0.5, 0.0, 0.5}, {0, 0.5, 0.5, 0.0}},
         1000,
         12.0},
        {"Si diamond / 2000 atoms / LCAO / semiconductor",
         10,
         5,
         5,
         2.4,
         2.4,
         2.4,
         1.25,
         1,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.0, 0.5, 0.5},
          {0, 0.5, 0.0, 0.5},
          {0, 0.5, 0.5, 0.0},
          {0, 0.25, 0.25, 0.25},
          {0, 0.25, 0.75, 0.75},
          {0, 0.75, 0.25, 0.75},
          {0, 0.75, 0.75, 0.25}},
         2000,
         4.0},
        {"NaCl / 3000 atoms / PW / ionic crystal",
         15,
         5,
         5,
         2.4,
         2.4,
         2.4,
         1.35,
         2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.0, 0.5, 0.5},
          {0, 0.5, 0.0, 0.5},
          {0, 0.5, 0.5, 0.0},
          {1, 0.5, 0.0, 0.0},
          {1, 0.0, 0.5, 0.0},
          {1, 0.0, 0.0, 0.5},
          {1, 0.5, 0.5, 0.5}},
         3000,
         6.0},
        {"TiO2 rutile / 4200 atoms / LCAO / complex oxide",
         10,
         10,
         7,
         2.4,
         2.4,
         1.56,
         1.25,
         2,
         {{0, 0.0, 0.0, 0.0},
          {0, 0.5, 0.5, 0.5},
          {1, 0.305, 0.305, 0.0},
          {1, 0.695, 0.695, 0.0},
          {1, 0.805, 0.195, 0.5},
          {1, 0.195, 0.805, 0.5}},
         4200,
         4.0},
    };

    std::ofstream ofs("sltk_material_runtime_runner.out");
    for (const auto& material : materials)
    {
        run_case(material, ofs);
    }
    return 0;
}
