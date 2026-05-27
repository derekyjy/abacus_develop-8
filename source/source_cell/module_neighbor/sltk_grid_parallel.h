#ifndef SLTK_GRID_PARALLEL_H
#define SLTK_GRID_PARALLEL_H

#include "sltk_grid.h"

#include <mpi.h>
#include <omp.h>
#include <chrono>
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
    void broadcast_atoms_data(MPI_Comm comm);

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
