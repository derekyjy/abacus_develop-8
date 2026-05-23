#include "../sltk_atom_arrange.h"

#define private public
#include "source_io/module_parameter/parameter.h"
#undef private
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "prepare_unitcell.h"
#include "source_cell/read_stru.h"
#ifdef __LCAO
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
LCAO_Orbitals::LCAO_Orbitals()
{
}
LCAO_Orbitals::~LCAO_Orbitals()
{
}
#endif
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

/************************************************
 *  unit test of atom_arrange
 ***********************************************/

/**
 * - Tested Functions:
 *   - atom_arrange::delete_vector(void)
 *     - delete vector
 *   - atom_arrange::set_sr_NL
 * 	   - set the sr: search radius including nonlocal beta
 *   - filter_adjs function
 *     - filter AdjacentAtomInfo to the minimized adjacent atoms
 */

void SetGlobalV()
{
    PARAM.input.test_grid = false;
}

namespace
{
struct SltkMaterialCase
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
    std::vector<int> type_counts_per_cell;
    std::vector<std::tuple<int, double, double, double>> basis;
    int expected_atoms;
    double expected_average_neighbors;
};

std::unique_ptr<UnitCell> make_sltk_crystal_case(const SltkMaterialCase& material)
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

void expect_sltk_material_result(const SltkMaterialCase& material, std::ofstream& ofs)
{
    ASSERT_EQ(material.type_counts_per_cell.size(), static_cast<size_t>(material.ntype)) << material.name;
    std::vector<int> actual_type_counts_per_cell(material.ntype, 0);
    for (const auto& atom_basis : material.basis)
    {
        ++actual_type_counts_per_cell[std::get<0>(atom_basis)];
    }
    ASSERT_EQ(actual_type_counts_per_cell, material.type_counts_per_cell) << material.name;

    std::unique_ptr<UnitCell> ucell = make_sltk_crystal_case(material);
    ASSERT_EQ(ucell->nat, material.expected_atoms) << material.name;
    int total_atoms = 0;
    for (int type = 0; type < ucell->ntype; ++type)
    {
        total_atoms += ucell->atoms[type].na;
    }
    ASSERT_EQ(total_atoms, material.expected_atoms) << material.name;

    Grid_Driver grid_d(PARAM.input.test_deconstructor, PARAM.input.test_grid);
    const auto start = std::chrono::steady_clock::now();
    grid_d.init(ofs, *ucell, material.cutoff, true);
    const auto finish = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(finish - start).count();

    long long total_neighbors = 0;
    int atoms_with_neighbors = 0;
    AdjacentAtomInfo adjs;
    for (int type = 0; type < ucell->ntype; ++type)
    {
        for (int atom = 0; atom < ucell->atoms[type].na; ++atom)
        {
            grid_d.Find_atom(*ucell, type, atom, &adjs);
            EXPECT_GE(adjs.adj_num, 0) << material.name;
            total_neighbors += adjs.adj_num;
            if (adjs.adj_num > 0)
            {
                ++atoms_with_neighbors;
            }
            ASSERT_EQ(adjs.ntype.size(), static_cast<size_t>(adjs.adj_num + 1)) << material.name;
            ASSERT_EQ(adjs.natom.size(), static_cast<size_t>(adjs.adj_num + 1)) << material.name;
            ASSERT_EQ(adjs.box.size(), static_cast<size_t>(adjs.adj_num + 1)) << material.name;
            EXPECT_EQ(adjs.ntype.back(), type) << material.name;
            EXPECT_EQ(adjs.natom.back(), atom) << material.name;
            EXPECT_EQ(adjs.box.back().x, 0) << material.name;
            EXPECT_EQ(adjs.box.back().y, 0) << material.name;
            EXPECT_EQ(adjs.box.back().z, 0) << material.name;
        }
    }

    EXPECT_EQ(atoms_with_neighbors, material.expected_atoms) << material.name;
    EXPECT_GT(total_neighbors, material.expected_atoms) << material.name;

    const double average_neighbors = static_cast<double>(total_neighbors) / material.expected_atoms;
    if (material.expected_average_neighbors > 0.0)
    {
        EXPECT_NEAR(average_neighbors, material.expected_average_neighbors, 1.0e-12) << material.name;
    }
    std::cout << "[sltk] " << material.name << ": atoms=" << material.expected_atoms
              << ", avg_neighbors=" << average_neighbors << ", build_ms=" << elapsed_ms << std::endl;
}
} // namespace

class SltkAtomArrangeTest : public testing::Test
{
  protected:
    UnitCell* ucell;
    UcellTestPrepare utp = UcellTestLib["Si"];
    std::ofstream ofs;
    std::ifstream ifs;
    bool pbc = true;
    double radius = ((8 + 5.01) * 2.0 + 0.01) / 10.2;
    int test_atom_in = 0;
    std::string output;
    void SetUp()
    {
        SetGlobalV();
        ucell = utp.SetUcellInfo();
    }
    void TearDown()
    {
        delete ucell;
    }
};

TEST_F(SltkAtomArrangeTest, setsrNL)
{
    atom_arrange test;
    const std::string teststring = "m";
    double rcutmax_Phi = 1;
    double rcutmax_Beta = 2;
    bool gamma_only_local = true;
    double test_sr = 0;
    std::ofstream ofs;
    ofs.open("./to_test_arrange.txt");
    test_sr = test.set_sr_NL(ofs, teststring, rcutmax_Phi, rcutmax_Beta, gamma_only_local);
    EXPECT_DOUBLE_EQ(test_sr, 2.001);

    gamma_only_local = false;
    test_sr = test.set_sr_NL(ofs, teststring, rcutmax_Phi, rcutmax_Beta, gamma_only_local);
    EXPECT_DOUBLE_EQ(test_sr, 6.001);

    const std::string teststring2 = "no";
    test_sr = test.set_sr_NL(ofs, teststring2, rcutmax_Phi, rcutmax_Beta, gamma_only_local);
    ofs.close();
    std::ifstream ifs;
    std::string test2, s;
    ifs.open("./to_test_arrange.txt");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("Orbital max radius cutoff (Bohr) = 1"));
    EXPECT_THAT(str, testing::HasSubstr("Nonlocal proj. max radius cutoff (Bohr) = 2"));
    ifs.close();
    //remove("./to_test_arrange");
}

TEST_F(SltkAtomArrangeTest, Search)
{
    unitcell::check_dtau(ucell->atoms,ucell->ntype, ucell->lat0, ucell->latvec);
    Grid_Driver grid_d(PARAM.input.test_deconstructor, PARAM.input.test_grid);
    ofs.open("test.out");
    bool test_only = true;
    atom_arrange::search(pbc, ofs, grid_d, *ucell, radius, test_atom_in, test_only);
    EXPECT_EQ(grid_d.getType(0),0);
    EXPECT_EQ(grid_d.getNatom(0), 1); // adjacent atom is 1
    ofs.close();
    ifs.open("test.out");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("search neighboring atoms done."));
    remove("test.out");
}

TEST_F(SltkAtomArrangeTest, Filteradjs)
{
    unitcell::check_dtau(ucell->atoms,ucell->ntype, ucell->lat0, ucell->latvec);
    Grid_Driver grid_d(PARAM.input.test_deconstructor, PARAM.input.test_grid);
    ofs.open("test.out");
    bool test_only = true;
    atom_arrange::search(pbc, ofs, grid_d, *ucell, radius, test_atom_in, test_only);
    EXPECT_EQ(grid_d.getType(0),0);
    EXPECT_EQ(grid_d.getNatom(0), 1); // adjacent atom is 1
    ofs.close();
    ifs.open("test.out");
    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_THAT(str, testing::HasSubstr("search neighboring atoms done."));
    remove("test.out");

    AdjacentAtomInfo adjs;
    grid_d.Find_atom(*ucell, ucell->atoms[0].tau[0], 0, 0, &adjs);
    EXPECT_EQ(adjs.adj_num, 0);
    // add one adjacent atom
    adjs.adj_num++;
    adjs.adjacent_tau.push_back(ModuleBase::Vector3<double>(0,0,0));
    adjs.box.push_back(ModuleBase::Vector3<int>(0,0,0));
    adjs.natom.push_back(1);
    adjs.ntype.push_back(0);
    EXPECT_EQ(adjs.adj_num, 1);
    // filter adjs to no adjacent status
    std::vector<bool> is_adjs(adjs.adj_num + 1, false);
    is_adjs[0] = true;
    filter_adjs(is_adjs, adjs);
    EXPECT_EQ(adjs.adj_num, 0);
}

TEST_F(SltkAtomArrangeTest, MaterialCoverageAndRuntime)
{
    const std::vector<SltkMaterialCase> materials = {
        {
            "Al fcc / 1000 atoms / PW / metal",
            10,
            5,
            5,
            2.0,
            2.0,
            2.0,
            1.5,
            1,
            {4},
            {
                {0, 0.0, 0.0, 0.0},
                {0, 0.0, 0.5, 0.5},
                {0, 0.5, 0.0, 0.5},
                {0, 0.5, 0.5, 0.0},
            },
            1000,
            12.0,
        },
        {
            "Si diamond / 2000 atoms / LCAO / semiconductor",
            10,
            5,
            5,
            2.4,
            2.4,
            2.4,
            1.25,
            1,
            {8},
            {
                {0, 0.0, 0.0, 0.0},
                {0, 0.0, 0.5, 0.5},
                {0, 0.5, 0.0, 0.5},
                {0, 0.5, 0.5, 0.0},
                {0, 0.25, 0.25, 0.25},
                {0, 0.25, 0.75, 0.75},
                {0, 0.75, 0.25, 0.75},
                {0, 0.75, 0.75, 0.25},
            },
            2000,
            4.0,
        },
        {
            "NaCl / 3000 atoms / PW / ionic crystal",
            15,
            5,
            5,
            2.4,
            2.4,
            2.4,
            1.35,
            2,
            {4, 4},
            {
                {0, 0.0, 0.0, 0.0},
                {0, 0.0, 0.5, 0.5},
                {0, 0.5, 0.0, 0.5},
                {0, 0.5, 0.5, 0.0},
                {1, 0.5, 0.0, 0.0},
                {1, 0.0, 0.5, 0.0},
                {1, 0.0, 0.0, 0.5},
                {1, 0.5, 0.5, 0.5},
            },
            3000,
            6.0,
        },
        {
            "TiO2 rutile / 4200 atoms / LCAO / complex oxide",
            10,
            10,
            7,
            2.4,
            2.4,
            1.56,
            1.25,
            2,
            {2, 4},
            {
                {0, 0.0, 0.0, 0.0},
                {0, 0.5, 0.5, 0.5},
                {1, 0.305, 0.305, 0.0},
                {1, 0.695, 0.695, 0.0},
                {1, 0.805, 0.195, 0.5},
                {1, 0.195, 0.805, 0.5},
            },
            4200,
            4.0,
        },
    };

    ofs.open("sltk_material_runtime.out");
    for (const auto& material : materials)
    {
        expect_sltk_material_result(material, ofs);
    }
    ofs.close();
    remove("sltk_material_runtime.out");
}
