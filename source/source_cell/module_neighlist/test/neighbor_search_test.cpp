#include <gtest/gtest.h>
#include "../neighbor_search.h"
#include "../unitcell_plus.h"

#include <numeric>
#include <string>
#include <tuple>

namespace
{
struct MaterialNeighborCase
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
};

UnitCellPlus make_crystal_case(const MaterialNeighborCase& material)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.ntype = material.ntype;
    ucell.na.assign(material.ntype, 0);
    ucell.latvec.e11 = material.nx * material.lattice_x;
    ucell.latvec.e12 = 0.0;
    ucell.latvec.e13 = 0.0;
    ucell.latvec.e21 = 0.0;
    ucell.latvec.e22 = material.ny * material.lattice_y;
    ucell.latvec.e23 = 0.0;
    ucell.latvec.e31 = 0.0;
    ucell.latvec.e32 = 0.0;
    ucell.latvec.e33 = material.nz * material.lattice_z;
    ucell.omega = ucell.latvec.e11 * ucell.latvec.e22 * ucell.latvec.e33;

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
        ucell.na[type] = static_cast<int>(atoms_by_type[type].size());
        ucell.tau.insert(ucell.tau.end(), atoms_by_type[type].begin(), atoms_by_type[type].end());
    }
    ucell.nat = static_cast<int>(ucell.tau.size());
    ucell.compute_naa();
    return ucell;
}

void expect_valid_neighbor_result(const MaterialNeighborCase& material)
{
    ASSERT_EQ(material.type_counts_per_cell.size(), static_cast<size_t>(material.ntype)) << material.name;
    std::vector<int> actual_type_counts_per_cell(material.ntype, 0);
    for (const auto& atom_basis : material.basis)
    {
        ++actual_type_counts_per_cell[std::get<0>(atom_basis)];
    }
    ASSERT_EQ(actual_type_counts_per_cell, material.type_counts_per_cell) << material.name;

    UnitCellPlus ucell = make_crystal_case(material);
    ASSERT_EQ(ucell.nat, material.expected_atoms) << material.name;
    ASSERT_EQ(std::accumulate(ucell.na.begin(), ucell.na.end(), 0), material.expected_atoms) << material.name;

    NeighborSearch ns;
    ns.init(ucell, material.cutoff, 0);
    ns.build_neighbors();

    NeighborList& list = ns.get_neighbor_list();
    ASSERT_EQ(ns.inside_atoms.size(), static_cast<size_t>(material.expected_atoms)) << material.name;
    ASSERT_EQ(list.nlocal, material.expected_atoms) << material.name;
    ASSERT_EQ(list.numneigh.size(), static_cast<size_t>(material.expected_atoms)) << material.name;
    ASSERT_EQ(list.firstneigh.size(), static_cast<size_t>(material.expected_atoms)) << material.name;

    int total_neighbors = 0;
    int atoms_with_neighbors = 0;
    for (int atom = 0; atom < material.expected_atoms; ++atom)
    {
        EXPECT_GE(list.numneigh[atom], 0) << material.name;
        total_neighbors += list.numneigh[atom];
        if (list.numneigh[atom] > 0)
        {
            ++atoms_with_neighbors;
            ASSERT_NE(list.firstneigh[atom], nullptr) << material.name;
        }
        for (int neigh = 0; neigh < list.numneigh[atom]; ++neigh)
        {
            EXPECT_NE(list.firstneigh[atom][neigh], ns.inside_atoms[atom].atom_id) << material.name;
        }
    }

    EXPECT_EQ(atoms_with_neighbors, material.expected_atoms) << material.name;
    EXPECT_GT(total_neighbors, material.expected_atoms) << material.name;
}
} // namespace

TEST(NeighborSearchTest, TwoAtomsNeighbor)
{
    UnitCellPlus ucell;

    ucell.lat0 = 1.0;
    ucell.omega = 1.0;

    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {2};
    ucell.nat = 2;

    ucell.tau = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0}
    };
    ucell.compute_naa();

    NeighborSearch ns;

    double cutoff = 1.0;

    ns.init(ucell, cutoff, 0);
    ns.build_neighbors();

    auto &list = ns.get_neighbor_list();

    ASSERT_EQ(list.numneigh.size(), 2);

    EXPECT_EQ(list.numneigh[0], 8);
    EXPECT_EQ(list.numneigh[1], 8);
}

TEST(NeighborSearchTest, NoNeighbor)
{
    UnitCellPlus ucell;

    ucell.lat0 = 1.0;
    ucell.omega = 1.0;

    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {2};
    ucell.nat = 2;

    ucell.tau = {
        {0.0, 0.0, 0.0},
        {5.0, 0.0, 0.0}
    };
    ucell.compute_naa();

    NeighborSearch ns;

    // use a smaller search radius to avoid counting periodic-image neighbors
    ns.init(ucell, 0.1, 0);
    ns.build_neighbors();

    auto &list = ns.get_neighbor_list();

    EXPECT_EQ(list.numneigh[0], 0);
    EXPECT_EQ(list.numneigh[1], 0);
}

TEST(NeighborSearchUnit, UCellToInputAtoms)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {2};
    ucell.nat = 2;
    ucell.tau = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0}
    };
    ucell.compute_naa();
    NeighborSearch ns;
    auto inputs = ns.ucell_to_input_atoms(ucell);

    EXPECT_EQ(inputs.n_atoms, 2);
    ASSERT_EQ(inputs.InputAtom.size(), 2);
    EXPECT_DOUBLE_EQ(inputs.InputAtom[0].position_x, 0.0);
    EXPECT_DOUBLE_EQ(inputs.InputAtom[1].position_x, 0.5);
    EXPECT_DOUBLE_EQ(inputs.x_low, 0.0);
    EXPECT_DOUBLE_EQ(inputs.x_high, 0.5);
}

TEST(NeighborSearchUnit, CheckExpandAndSetMembers)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {2};
    ucell.nat = 2;
    ucell.tau = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0}
    };
    ucell.compute_naa();
    NeighborSearch ns;
    ns.search_radius = 1.0; // use search radius = 1 for Check_Expand_Condition
    ns.Check_Expand_Condition(ucell);

    // For identity lattice with search_radius=1 expected ceil produce values
    EXPECT_EQ(ns.glayerX, 2);
    EXPECT_EQ(ns.glayerY, 2);
    EXPECT_EQ(ns.glayerZ, 2);
    EXPECT_EQ(ns.glayerX_minus, 1);

    // Now populate all_atoms and check count
    ns.setMemberVariables(ucell);
    int images_x = ns.glayerX + ns.glayerX_minus; // iterations in x
    int images_y = ns.glayerY + ns.glayerY_minus;
    int images_z = ns.glayerZ + ns.glayerZ_minus;
    int expected = images_x * images_y * images_z * 2; // 2 atoms per cell
    EXPECT_EQ(static_cast<int>(ns.all_atoms.size()), expected);
}

TEST(NeighborSearchUnit, DistanceBox)
{
    NeighborSearch ns;
    // set a single cell region at x=0..1,y=0..1,z=0..1
    ns.x = 0; ns.y = 0; ns.z = 0;
    ns.wide_x = 1.0; ns.wide_y = 1.0; ns.wide_z = 1.0;

    double inside = ns.distance(0.2, 0.5, 0.5, 0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(inside, 0.0);

    double outside = ns.distance(2.0, 0.5, 0.5, 0.0, 0.0, 0.0);
    // squared distance should be (2-1)^2 = 1
    EXPECT_DOUBLE_EQ(outside, 1.0);
}

TEST(NeighborSearchUnit, DecomposeCases)
{
    NeighborSearch ns;
    int nx, ny, nz;

    ns.decompose(8, nx, ny, nz);
    EXPECT_EQ(nx * ny * nz, 8);
    // expect somewhat balanced cube factors for 8
    EXPECT_EQ(nx, 2);
    EXPECT_EQ(ny, 2);
    EXPECT_EQ(nz, 2);

    ns.decompose(7, nx, ny, nz);
    EXPECT_EQ(nx * ny * nz, 7);
    EXPECT_EQ(nx, 1);
    EXPECT_EQ(ny, 1);
    EXPECT_EQ(nz, 7);
}

TEST(NeighborSearchUnit, UCellToInputAtomsMultipleTypes)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 2;
    ucell.na = {1, 2};
    ucell.nat = 3;
    ucell.tau = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0}
    };
    ucell.compute_naa();
    NeighborSearch ns;
    auto inputs = ns.ucell_to_input_atoms(ucell);

    EXPECT_EQ(inputs.n_atoms, 3);
    ASSERT_EQ(inputs.InputAtom.size(), 3);
    EXPECT_DOUBLE_EQ(inputs.InputAtom[2].position_y, 0.5);
}

TEST(NeighborSearchUnit, DecomposePrimeNumber)
{
    NeighborSearch ns;
    int nx, ny, nz;
    ns.decompose(13, nx, ny, nz);
    EXPECT_EQ(nx * ny * nz, 13);
    EXPECT_EQ(nx, 1);
    EXPECT_EQ(ny, 1);
    EXPECT_EQ(nz, 13);
}

TEST(NeighborSearchUnit, NonOrthogonalLatticeExpand)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    // skewed lattice
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0.3; ucell.latvec.e13 = 0.0;
    ucell.latvec.e21 = 0.1; ucell.latvec.e22 = 1.0; ucell.latvec.e23 = 0.0;
    ucell.latvec.e31 = 0.0; ucell.latvec.e32 = 0.0; ucell.latvec.e33 = 1.0;

    ucell.ntype = 1;
    ucell.na = {1};
    ucell.nat = 1;
    ucell.tau = {{0.0, 0.0, 0.0}};
    ucell.compute_naa();

    NeighborSearch ns;
    ns.search_radius = 2.5;
    ns.Check_Expand_Condition(ucell);
    // for skewed lattice, expansion layers should be >= 1
    EXPECT_GE(ns.glayerX, 1);
    EXPECT_GE(ns.glayerY, 1);
    EXPECT_GE(ns.glayerZ, 1);
}

// --- additional tests to cover remaining branches in neighbor_search.cpp ---

TEST(NeighborSearchInit_WideZero_CentralInside, SingleAtomCell)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {1};
    ucell.nat = 1;
    ucell.tau = {{0.0, 0.0, 0.0}};
    ucell.compute_naa();

    NeighborSearch ns;
    // choose sr small enough; with mpi_size fixed to 1 in init, wide_* become 0
    ns.init(ucell, 0.1, 0);
    // central cell atom should be counted as inside
    EXPECT_EQ(ns.inside_atoms.size(), 1);
    EXPECT_EQ(ns.neighbor_list.nlocal, static_cast<int>(ns.inside_atoms.size()));
}

TEST(NeighborSearchInit_MpiRankIndexing, RankValues)
{
    UnitCellPlus ucell;
    ucell.lat0 = 1.0;
    ucell.omega = 1.0;
    ucell.latvec.e11 = 1; ucell.latvec.e12 = 0; ucell.latvec.e13 = 0;
    ucell.latvec.e21 = 0; ucell.latvec.e22 = 1; ucell.latvec.e23 = 0;
    ucell.latvec.e31 = 0; ucell.latvec.e32 = 0; ucell.latvec.e33 = 1;

    ucell.ntype = 1;
    ucell.na = {1};
    ucell.nat = 1;
    ucell.tau = {{0.0, 0.0, 0.0}};
    ucell.compute_naa();

    NeighborSearch ns0;
    ns0.init(ucell, 0.5, 0);
    // with mpi_size fixed to 1 in init, nx=ny=nz=1; for rank 0 expect x=y=0,z=0
    EXPECT_EQ(ns0.x, 0);
    EXPECT_EQ(ns0.y, 0);
    EXPECT_EQ(ns0.z, 0);

}

TEST(NeighborSearchDistance_OutsideCases, VariousAxes)
{
    NeighborSearch ns;
    ns.x = 0; ns.y = 0; ns.z = 0;
    ns.wide_x = 2.0; ns.wide_y = 3.0; ns.wide_z = 4.0;

    // position inside box along x (no dx), but outside along y by above high bound
    double d = ns.distance(0.5, 4.5, 1.0, 0.0, 0.0, 0.0);
    // dy = position_y - (y_low + (y+1)*wide_y) = 4.5 - 3.0 = 1.5 -> squared 2.25
    // dx = 0, dz = 0 -> total 2.25
    EXPECT_DOUBLE_EQ(d, 2.25);

    // position left of low bound on x
    double d2 = ns.distance(-1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
    // dx = x_low - position_x = 0 - (-1) = 1 -> squared 1
    EXPECT_DOUBLE_EQ(d2, 1.0);
}

TEST(NeighborSearchDecompose_SmallSizes, TwoAndOne)
{
    NeighborSearch ns;
    int nx, ny, nz;
    ns.decompose(2, nx, ny, nz);
    EXPECT_EQ(nx * ny * nz, 2);
    // possible decomposition is nx=1, ny=1, nz=2 (or nx=1, ny=2, nz=1 depending on algorithm)
    EXPECT_EQ(nx, 1);

    ns.decompose(1, nx, ny, nz);
    EXPECT_EQ(nx, 1);
    EXPECT_EQ(ny, 1);
    EXPECT_EQ(nz, 1);
}

TEST(NeighborSearchMaterialCoverage, BenchmarkSystemsFromProjectReport)
{
    const std::vector<MaterialNeighborCase> materials = {
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
        },
    };

    for (const auto& material : materials)
    {
        expect_valid_neighbor_result(material);
    }
}

// end of additional tests
