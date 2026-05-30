#ifndef SLTK_GRID_PARALLEL_H
#define SLTK_GRID_PARALLEL_H

#include "sltk_grid.h"

#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <array>
#include <vector>
#include <utility>

class GridParallel : public Grid
{
  public:
    GridParallel();
    GridParallel(const int& test_grid_in);
    ~GridParallel();

    using Grid::init;

    double Construct_Adjacent_parallel(const UnitCell& ucell, MPI_Comm comm = MPI_COMM_WORLD);

    double Construct_Adjacent_serial(const UnitCell& ucell);

    bool compare_adj_info(const Grid& other) const;

  private:
    struct DomainDecomposition
    {
        int px;
        int py;
        int pz;
    };

    struct DomainBounds
    {
        int x_begin;
        int x_end;
        int y_begin;
        int y_end;
        int z_begin;
        int z_end;
    };

    DomainDecomposition choose_domain_decomposition(int mpi_size) const;

    DomainBounds rank_domain_bounds(int rank, const DomainDecomposition& decomp) const;

    bool atom_in_domain(const FAtom& atom, const DomainBounds& bounds);

    std::array<int, 3> rank_domain_coord(int rank, const DomainDecomposition& decomp) const;

    int domain_rank_from_coord(int x, int y, int z, const DomainDecomposition& decomp) const;

    int atom_box_x(const FAtom& atom) const;

    int atom_box_y(const FAtom& atom) const;

    int atom_box_z(const FAtom& atom) const;

    bool atom_in_ghost_layer(const FAtom& atom,
                             const DomainBounds& bounds,
                             int dx,
                             int dy,
                             int dz,
                             int search_span) const;

    std::vector<FAtom> exchange_ghost_atoms(const std::vector<FAtom>& owned_atoms,
                                            const DomainBounds& bounds,
                                            const DomainDecomposition& decomp,
                                            MPI_Comm comm) const;

    void rebuild_local_search_grid(const std::vector<FAtom>& owned_atoms,
                                   const std::vector<FAtom>& ghost_atoms);

    struct NeighborEntry
    {
        int type;
        int natom;
        int cell_x;
        int cell_y;
        int cell_z;
        double x;
        double y;
        double z;
    };

    void serialize_neighbors(int i_type, int j_atom, std::vector<NeighborEntry>& entries) const;

    void deserialize_neighbors(int i_type, int j_atom, const std::vector<NeighborEntry>& entries);

    std::vector<FAtom> flat_atoms;
};

#endif
