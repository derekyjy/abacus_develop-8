#include "sltk_grid_parallel.h"

#include "source_base/global_function.h"
#include "source_base/timer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>

GridParallel::GridParallel() : Grid()
{
}

GridParallel::GridParallel(const int& test_grid_in) : Grid(test_grid_in)
{
}

GridParallel::~GridParallel()
{
}

double GridParallel::Construct_Adjacent_serial(const UnitCell& ucell)
{
    double t_start = omp_get_wtime();

    for (int i_type = 0; i_type < ucell.ntype; i_type++)
    {
        for (int j_atom = 0; j_atom < ucell.atoms[i_type].na; j_atom++)
        {
            FAtom atom(ucell.atoms[i_type].tau[j_atom].x,
                       ucell.atoms[i_type].tau[j_atom].y,
                       ucell.atoms[i_type].tau[j_atom].z,
                       i_type,
                       j_atom,
                       0, 0, 0);
            this->Construct_Adjacent_near_box(atom);
        }
    }

    double t_end = omp_get_wtime();
    return t_end - t_start;
}

void GridParallel::broadcast_atoms_data(MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int total_atoms = 0;
    std::vector<int> box_counts;
    std::vector<double> atom_data;

    if (rank == 0)
    {
        for (int bx = 0; bx < box_nx; bx++)
        {
            for (int by = 0; by < box_ny; by++)
            {
                for (int bz = 0; bz < box_nz; bz++)
                {
                    int count = static_cast<int>(atoms_in_box[bx][by][bz].size());
                    box_counts.push_back(count);
                    total_atoms += count;
                    for (const auto& atom : atoms_in_box[bx][by][bz])
                    {
                        atom_data.push_back(atom.x);
                        atom_data.push_back(atom.y);
                        atom_data.push_back(atom.z);
                        atom_data.push_back(static_cast<double>(atom.type));
                        atom_data.push_back(static_cast<double>(atom.natom));
                        atom_data.push_back(static_cast<double>(atom.cell_x));
                        atom_data.push_back(static_cast<double>(atom.cell_y));
                        atom_data.push_back(static_cast<double>(atom.cell_z));
                    }
                }
            }
        }
    }

    int num_boxes = box_nx * box_ny * box_nz;
    MPI_Bcast(&num_boxes, 1, MPI_INT, 0, comm);

    if (rank != 0)
    {
        box_counts.resize(num_boxes);
    }
    MPI_Bcast(box_counts.data(), num_boxes, MPI_INT, 0, comm);

    MPI_Bcast(&total_atoms, 1, MPI_INT, 0, comm);

    if (rank != 0)
    {
        atom_data.resize(total_atoms * 8);
    }
    MPI_Bcast(atom_data.data(), total_atoms * 8, MPI_DOUBLE, 0, comm);

    if (rank != 0)
    {
        flat_atoms.clear();
        atoms_in_box.clear();
        atoms_in_box.resize(box_nx);
        for (int i = 0; i < box_nx; i++)
        {
            atoms_in_box[i].resize(box_ny);
            for (int j = 0; j < box_ny; j++)
            {
                atoms_in_box[i][j].resize(box_nz);
            }
        }

        int idx = 0;
        int data_idx = 0;
        for (int bx = 0; bx < box_nx; bx++)
        {
            for (int by = 0; by < box_ny; by++)
            {
                for (int bz = 0; bz < box_nz; bz++)
                {
                    int count = box_counts[idx++];
                    for (int c = 0; c < count; c++)
                    {
                        double ax = atom_data[data_idx++];
                        double ay = atom_data[data_idx++];
                        double az = atom_data[data_idx++];
                        int atype = static_cast<int>(atom_data[data_idx++]);
                        int anatom = static_cast<int>(atom_data[data_idx++]);
                        int acx = static_cast<int>(atom_data[data_idx++]);
                        int acy = static_cast<int>(atom_data[data_idx++]);
                        int acz = static_cast<int>(atom_data[data_idx++]);

                        FAtom fatom(ax, ay, az, atype, anatom, acx, acy, acz);
                        atoms_in_box[bx][by][bz].push_back(fatom);
                        flat_atoms.push_back(fatom);
                    }
                }
            }
        }
    }
    else
    {
        flat_atoms.clear();
        for (int bx = 0; bx < box_nx; bx++)
        {
            for (int by = 0; by < box_ny; by++)
            {
                for (int bz = 0; bz < box_nz; bz++)
                {
                    for (const auto& atom : atoms_in_box[bx][by][bz])
                    {
                        flat_atoms.push_back(atom);
                    }
                }
            }
        }
    }

    all_adj_info.clear();
    all_adj_info.resize(ucell_ptr->ntype);
    for (int i = 0; i < ucell_ptr->ntype; i++)
    {
        all_adj_info[i].resize(ucell_ptr->atoms[i].na);
    }
}

double GridParallel::Construct_Adjacent_parallel(const UnitCell& ucell, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double t_start = MPI_Wtime();

    broadcast_atoms_data(comm);

    std::vector<std::pair<int, int>> atom_pairs;
    if (rank == 0)
    {
        for (int i_type = 0; i_type < ucell.ntype; i_type++)
        {
            for (int j_atom = 0; j_atom < ucell.atoms[i_type].na; j_atom++)
            {
                atom_pairs.push_back({i_type, j_atom});
            }
        }
    }

    int total_atoms = 0;
    if (rank == 0)
    {
        total_atoms = static_cast<int>(atom_pairs.size());
    }
    MPI_Bcast(&total_atoms, 1, MPI_INT, 0, comm);

    int base = total_atoms / size;
    int remainder = total_atoms % size;
    int my_start = rank * base + std::min(rank, remainder);
    int my_end = my_start + base + (rank < remainder ? 1 : 0);

    int my_count = my_end - my_start;

    std::vector<int> all_starts(size);
    std::vector<int> all_counts(size);
    int local_start_int[2] = {my_start, my_count};
    MPI_Allgather(local_start_int, 2, MPI_INT, all_starts.data(), 2, MPI_INT,
                  comm);

    double t_comp_start = MPI_Wtime();

#pragma omp parallel
    {
        for (int local_idx = my_start + omp_get_thread_num();
             local_idx < my_end;
             local_idx += omp_get_num_threads())
        {
            int global_idx = local_idx;
            int i_type = 0;
            int j_atom = 0;
            int acc = 0;
            for (int it = 0; it < ucell.ntype; it++)
            {
                if (global_idx < acc + ucell.atoms[it].na)
                {
                    i_type = it;
                    j_atom = global_idx - acc;
                    break;
                }
                acc += ucell.atoms[it].na;
            }

            FAtom atom(ucell.atoms[i_type].tau[j_atom].x,
                       ucell.atoms[i_type].tau[j_atom].y,
                       ucell.atoms[i_type].tau[j_atom].z,
                       i_type,
                       j_atom,
                       0, 0, 0);
            this->Construct_Adjacent_near_box(atom);
        }
    }

    double t_comp_end = MPI_Wtime();

    std::vector<std::vector<NeighborEntry>> all_serialized;
    if (rank == 0)
    {
        all_serialized.resize(total_atoms);
    }

    int send_count = 0;
    std::vector<std::vector<NeighborEntry>> my_entries(my_count);
    for (int local_idx = my_start; local_idx < my_end; local_idx++)
    {
        int global_idx = local_idx;
        int i_type = 0;
        int j_atom = 0;
        int acc = 0;
        for (int it = 0; it < ucell.ntype; it++)
        {
            if (global_idx < acc + ucell.atoms[it].na)
            {
                i_type = it;
                j_atom = global_idx - acc;
                break;
            }
            acc += ucell.atoms[it].na;
        }
        serialize_neighbors(i_type, j_atom, my_entries[local_idx - my_start]);
        send_count += static_cast<int>(my_entries[local_idx - my_start].size());
    }

    std::vector<int> size_buf(my_count);
    int idx_buf = 0;
    for (int i = 0; i < my_count; i++)
    {
        size_buf[i] = static_cast<int>(my_entries[i].size());
    }

    int total_entries = 0;
    MPI_Reduce(&send_count, &total_entries, 1, MPI_INT, MPI_SUM, 0, comm);

    double* data_buf = new double[send_count * 8];
    int pos = 0;
    for (int i = 0; i < my_count; i++)
    {
        for (const auto& e : my_entries[i])
        {
            data_buf[pos++] = static_cast<double>(e.type);
            data_buf[pos++] = static_cast<double>(e.natom);
            data_buf[pos++] = static_cast<double>(e.cell_x);
            data_buf[pos++] = static_cast<double>(e.cell_y);
            data_buf[pos++] = static_cast<double>(e.cell_z);
            data_buf[pos++] = e.x;
            data_buf[pos++] = e.y;
            data_buf[pos++] = e.z;
        }
    }

    int* recv_counts = nullptr;
    int* recv_displs = nullptr;
    double* recv_buf = nullptr;

    if (rank == 0)
    {
        recv_counts = new int[size];
        recv_displs = new int[size];
    }

    MPI_Gather(&send_count, 1, MPI_INT,
               rank == 0 ? recv_counts : nullptr, 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        recv_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }
        recv_buf = new double[total_entries * 8];
    }

    MPI_Gatherv(data_buf, send_count * 8, MPI_DOUBLE,
                rank == 0 ? recv_buf : nullptr,
                rank == 0 ? recv_counts : nullptr,
                rank == 0 ? recv_displs : nullptr,
                MPI_DOUBLE, 0, comm);

    delete[] data_buf;

    int* size_recv_counts = nullptr;
    int* size_recv_displs = nullptr;
    int* size_recv_buf = nullptr;

    if (rank == 0)
    {
        size_recv_counts = new int[size];
        size_recv_displs = new int[size];
    }

    MPI_Gather(&my_count, 1, MPI_INT,
               rank == 0 ? size_recv_counts : nullptr, 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        size_recv_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            size_recv_displs[i] = size_recv_displs[i - 1] + size_recv_counts[i - 1];
        }
        size_recv_buf = new int[total_atoms];
    }

    MPI_Gatherv(size_buf.data(), my_count, MPI_INT,
                rank == 0 ? size_recv_buf : nullptr,
                rank == 0 ? size_recv_counts : nullptr,
                rank == 0 ? size_recv_displs : nullptr,
                MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_idx = 0;
        for (int p = 0; p < size; p++)
        {
            int count = size_recv_counts[p];
            for (int i = 0; i < count; i++)
            {
                int num_entries = size_recv_buf[size_recv_displs[p] + i];
                std::vector<NeighborEntry> entries;
                entries.reserve(num_entries);
                for (int e = 0; e < num_entries; e++)
                {
                    int entry_idx = (recv_displs[p] + total_idx) * 8;
                    NeighborEntry entry;
                    entry.type = static_cast<int>(recv_buf[entry_idx]);
                    entry.natom = static_cast<int>(recv_buf[entry_idx + 1]);
                    entry.cell_x = static_cast<int>(recv_buf[entry_idx + 2]);
                    entry.cell_y = static_cast<int>(recv_buf[entry_idx + 3]);
                    entry.cell_z = static_cast<int>(recv_buf[entry_idx + 4]);
                    entry.x = recv_buf[entry_idx + 5];
                    entry.y = recv_buf[entry_idx + 6];
                    entry.z = recv_buf[entry_idx + 7];
                    entries.push_back(entry);
                    total_idx++;
                }

                int global_atom_idx = size_recv_displs[p] + i;
                int i_type = 0;
                int j_atom = 0;
                int acc = 0;
                for (int it = 0; it < ucell.ntype; it++)
                {
                    if (global_atom_idx < acc + ucell.atoms[it].na)
                    {
                        i_type = it;
                        j_atom = global_atom_idx - acc;
                        break;
                    }
                    acc += ucell.atoms[it].na;
                }
                deserialize_neighbors(i_type, j_atom, entries);
            }
        }

        delete[] recv_counts;
        delete[] recv_displs;
        delete[] recv_buf;
        delete[] size_recv_counts;
        delete[] size_recv_displs;
        delete[] size_recv_buf;
    }

    double t_end = MPI_Wtime();

    return t_end - t_start;
}

void GridParallel::serialize_neighbors(int i_type, int j_atom,
                                       std::vector<NeighborEntry>& entries) const
{
    const auto& neighbors = all_adj_info[i_type][j_atom];
    entries.reserve(neighbors.size());
    for (const FAtom* atom : neighbors)
    {
        NeighborEntry e;
        e.type = atom->type;
        e.natom = atom->natom;
        e.cell_x = atom->cell_x;
        e.cell_y = atom->cell_y;
        e.cell_z = atom->cell_z;
        e.x = atom->x;
        e.y = atom->y;
        e.z = atom->z;
        entries.push_back(e);
    }
}

void GridParallel::deserialize_neighbors(int i_type, int j_atom,
                                         const std::vector<NeighborEntry>& entries)
{
    for (const auto& e : entries)
    {
        for (const auto& atom : flat_atoms)
        {
            if (atom.type == e.type && atom.natom == e.natom
                && atom.cell_x == e.cell_x && atom.cell_y == e.cell_y
                && atom.cell_z == e.cell_z)
            {
                all_adj_info[i_type][j_atom].push_back(
                    const_cast<FAtom*>(&atom));
                break;
            }
        }
    }
}

bool GridParallel::compare_adj_info(const Grid& other) const
{
    if (all_adj_info.size() != other.all_adj_info.size())
    {
        return false;
    }
    for (size_t i = 0; i < all_adj_info.size(); i++)
    {
        if (all_adj_info[i].size() != other.all_adj_info[i].size())
        {
            return false;
        }
        for (size_t j = 0; j < all_adj_info[i].size(); j++)
        {
            if (all_adj_info[i][j].size() != other.all_adj_info[i][j].size())
            {
                return false;
            }
            for (size_t k = 0; k < all_adj_info[i][j].size(); k++)
            {
                const FAtom* a1 = all_adj_info[i][j][k];
                const FAtom* a2 = other.all_adj_info[i][j][k];
                if (a1->type != a2->type || a1->natom != a2->natom
                    || a1->cell_x != a2->cell_x || a1->cell_y != a2->cell_y
                    || a1->cell_z != a2->cell_z)
                {
                    return false;
                }
            }
        }
    }
    return true;
}
