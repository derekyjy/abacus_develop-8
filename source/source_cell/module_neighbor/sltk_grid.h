#ifndef GRID_H
#define GRID_H

#include "source_cell/unitcell.h"
#include "sltk_atom.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

typedef std::vector<FAtom> AtomMap;

class Grid
{
  public:
    // Constructors and destructor
    // Grid is Global class,so init it with constant number
    Grid() : test_grid(0){};
    Grid(const int& test_grid_in);
    virtual ~Grid();

    Grid& operator=(Grid&&) = default;

    void init(std::ofstream& ofs, const UnitCell& ucell, const double radius_in, const bool boundary = true);

    // Data
    bool pbc=false; // When pbc is set to false, periodic boundary conditions are explicitly ignored.
    double sradius2=0.0; // searching radius squared (unit:lat0)
    double sradius=0.0;  // searching radius (unit:lat0)
    
    // coordinate range of the input atom (unit:lat0)
    double x_min=0.0;
    double y_min=0.0;
    double z_min=0.0;
    double x_max=0.0;
    double y_max=0.0;
    double z_max=0.0;

    // The algorithm for searching neighboring atoms uses a "box" partitioning method. 
    // Each box has an edge length of sradius, and the number of boxes in each direction is recorded here.
    double box_edge_length=0.0;
    int box_nx=0;
    int box_ny=0;
    int box_nz=0;

    void getBox(int& bx, int& by, int& bz, const double& x, const double& y, const double& z) const
    {
        bx = std::floor((x - x_min) / box_edge_length);
        by = std::floor((y - y_min) / box_edge_length);
        bz = std::floor((z - z_min) / box_edge_length);
    }
    // Stores the atoms after box partitioning.
    std::vector<std::vector<std::vector<AtomMap>>> atoms_in_box;

    struct BoxBounds
    {
        double x_min = std::numeric_limits<double>::max();
        double y_min = std::numeric_limits<double>::max();
        double z_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        double y_max = std::numeric_limits<double>::lowest();
        double z_max = std::numeric_limits<double>::lowest();
        bool empty = true;

        void add_atom(const FAtom& atom)
        {
            x_min = std::min(x_min, atom.x);
            y_min = std::min(y_min, atom.y);
            z_min = std::min(z_min, atom.z);
            x_max = std::max(x_max, atom.x);
            y_max = std::max(y_max, atom.y);
            z_max = std::max(z_max, atom.z);
            empty = false;
        }
    };

    // Per-box coordinate bounds used to skip boxes that cannot contain atoms
    // within the current search radius.
    std::vector<std::vector<std::vector<BoxBounds>>> box_bounds;

    // Stores the adjacent information of atoms. [ntype][natom][adj list]
    std::vector<std::vector< std::vector<FAtom *> >> all_adj_info;
    void clear_atoms()
    {
        // we have to clear the all_adj_info
        // because the pointers point to the memory in vector atoms_in_box
        all_adj_info.clear();

        atoms_in_box.clear();
        box_bounds.clear();
    }
    void clear_adj_info()
    {
        // here dont need to free the memory, 
        // because the pointers point to the memory in vector atoms_in_box
        all_adj_info.clear();
    }
    int getGlayerX() const
    {
        return glayerX;
    }
    int getGlayerY() const
    {
        return glayerY;
    }
    int getGlayerZ() const
    {
        return glayerZ;
    }
    int getGlayerX_minus() const
    {
        return glayerX_minus;
    }
    int getGlayerY_minus() const
    {
        return glayerY_minus;
    }
    int getGlayerZ_minus() const
    {
        return glayerZ_minus;
    }

    void Construct_Adjacent_omp(const UnitCell& ucell);

  protected:
    void Construct_Adjacent_near_box_local(const FAtom& fatom);
    void Construct_Adjacent_final(const FAtom& fatom1, FAtom* fatom2);

  private:
    int test_grid;

    void setMemberVariables(std::ofstream& ofs_in, const UnitCell& ucell);

    void Construct_Adjacent(const UnitCell& ucell);
    void Construct_Adjacent_near_box(const FAtom& fatom);

    void Check_Expand_Condition(const UnitCell& ucell);
    bool box_may_contain_neighbor(const FAtom& fatom, const BoxBounds& bounds) const;
    int glayerX=0;
    int glayerX_minus=0;
    int glayerY=0;
    int glayerY_minus=0;
    int glayerZ=0;
    int glayerZ_minus=0;
};

#endif
