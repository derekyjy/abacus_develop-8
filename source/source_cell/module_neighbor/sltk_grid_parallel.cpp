#include "sltk_grid_parallel.h"

#include "source_base/global_function.h"
#include "source_base/timer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

namespace
{
constexpr int FATOM_PACK_SIZE = 8;

void pack_atoms(const std::vector<FAtom>& atoms, std::vector<double>& buffer)
{
    buffer.clear();
    buffer.reserve(atoms.size() * FATOM_PACK_SIZE);
    for (const auto& atom : atoms)
    {
        buffer.push_back(atom.x);
        buffer.push_back(atom.y);
        buffer.push_back(atom.z);
        buffer.push_back(static_cast<double>(atom.type));
        buffer.push_back(static_cast<double>(atom.natom));
        buffer.push_back(static_cast<double>(atom.cell_x));
        buffer.push_back(static_cast<double>(atom.cell_y));
        buffer.push_back(static_cast<double>(atom.cell_z));
    }
}

void unpack_atoms(const std::vector<double>& buffer, std::vector<FAtom>& atoms)
{
    atoms.clear();
    atoms.reserve(buffer.size() / FATOM_PACK_SIZE);
    for (size_t i = 0; i + FATOM_PACK_SIZE - 1 < buffer.size(); i += FATOM_PACK_SIZE)
    {
        atoms.emplace_back(buffer[i],
                           buffer[i + 1],
                           buffer[i + 2],
                           static_cast<int>(buffer[i + 3]),
                           static_cast<int>(buffer[i + 4]),
                           static_cast<int>(buffer[i + 5]),
                           static_cast<int>(buffer[i + 6]),
                           static_cast<int>(buffer[i + 7]));
    }
}
} // namespace

GridParallel::GridParallel() : Grid()
{
}

GridParallel::GridParallel(const int& test_grid_in) : Grid(test_grid_in)
{
}

GridParallel::~GridParallel()
{
}

GridParallel::DomainDecomposition GridParallel::choose_domain_decomposition(int mpi_size) const
{
    DomainDecomposition best{1, 1, mpi_size};
    double best_score = std::numeric_limits<double>::max();

    for (int px = 1; px <= mpi_size; px++)
    {
        if (mpi_size % px != 0)
        {
            continue;
        }
        const int yz = mpi_size / px;
        for (int py = 1; py <= yz; py++)
        {
            if (yz % py != 0)
            {
                continue;
            }
            const int pz = yz / py;
            const double sx = static_cast<double>(std::max(1, box_nx)) / px;
            const double sy = static_cast<double>(std::max(1, box_ny)) / py;
            const double sz = static_cast<double>(std::max(1, box_nz)) / pz;
            const double max_side = std::max(sx, std::max(sy, sz));
            const double min_side = std::max(1.0, std::min(sx, std::min(sy, sz)));
            const double surface = sx * sy + sx * sz + sy * sz;
            const double score = max_side / min_side + 1.0e-6 * surface;

            if (score < best_score)
            {
                best = {px, py, pz};
                best_score = score;
            }
        }
    }

    return best;
}

GridParallel::DomainBounds GridParallel::rank_domain_bounds(
    int rank,
    const DomainDecomposition& decomp) const
{
    const int rx = rank % decomp.px;
    const int ry = (rank / decomp.px) % decomp.py;
    const int rz = rank / (decomp.px * decomp.py);

    const auto split_begin = [](int n, int p, int coord) {
        return coord * (n / p) + std::min(coord, n % p);
    };
    const auto split_end = [&](int n, int p, int coord) {
        return split_begin(n, p, coord + 1);
    };

    return {split_begin(box_nx, decomp.px, rx),
            split_end(box_nx, decomp.px, rx),
            split_begin(box_ny, decomp.py, ry),
            split_end(box_ny, decomp.py, ry),
            split_begin(box_nz, decomp.pz, rz),
            split_end(box_nz, decomp.pz, rz)};
}

bool GridParallel::atom_in_domain(const FAtom& atom, const DomainBounds& bounds)
{
    int bx = 0;
    int by = 0;
    int bz = 0;
    getBox(bx, by, bz, atom.x, atom.y, atom.z);

    bx = std::max(0, std::min(box_nx - 1, bx));
    by = std::max(0, std::min(box_ny - 1, by));
    bz = std::max(0, std::min(box_nz - 1, bz));

    return bx >= bounds.x_begin && bx < bounds.x_end
           && by >= bounds.y_begin && by < bounds.y_end
           && bz >= bounds.z_begin && bz < bounds.z_end;
}

std::array<int, 3> GridParallel::rank_domain_coord(
    int rank,
    const DomainDecomposition& decomp) const
{
    return {rank % decomp.px,
            (rank / decomp.px) % decomp.py,
            rank / (decomp.px * decomp.py)};
}

int GridParallel::domain_rank_from_coord(
    int x,
    int y,
    int z,
    const DomainDecomposition& decomp) const
{
    if (x < 0 || x >= decomp.px || y < 0 || y >= decomp.py || z < 0 || z >= decomp.pz)
    {
        return MPI_PROC_NULL;
    }
    return x + y * decomp.px + z * decomp.px * decomp.py;
}

int GridParallel::atom_box_x(const FAtom& atom) const
{
    int bx = 0;
    int by = 0;
    int bz = 0;
    getBox(bx, by, bz, atom.x, atom.y, atom.z);
    return std::max(0, std::min(box_nx - 1, bx));
}

int GridParallel::atom_box_y(const FAtom& atom) const
{
    int bx = 0;
    int by = 0;
    int bz = 0;
    getBox(bx, by, bz, atom.x, atom.y, atom.z);
    return std::max(0, std::min(box_ny - 1, by));
}

int GridParallel::atom_box_z(const FAtom& atom) const
{
    int bx = 0;
    int by = 0;
    int bz = 0;
    getBox(bx, by, bz, atom.x, atom.y, atom.z);
    return std::max(0, std::min(box_nz - 1, bz));
}

bool GridParallel::atom_in_ghost_layer(const FAtom& atom,
                                       const DomainBounds& bounds,
                                       int dx,
                                       int dy,
                                       int dz,
                                       int search_span) const
{
    const int bx = atom_box_x(atom);
    const int by = atom_box_y(atom);
    const int bz = atom_box_z(atom);

    if (dx < 0 && bx >= bounds.x_begin + search_span)
    {
        return false;
    }
    if (dx > 0 && bx < bounds.x_end - search_span)
    {
        return false;
    }
    if (dy < 0 && by >= bounds.y_begin + search_span)
    {
        return false;
    }
    if (dy > 0 && by < bounds.y_end - search_span)
    {
        return false;
    }
    if (dz < 0 && bz >= bounds.z_begin + search_span)
    {
        return false;
    }
    if (dz > 0 && bz < bounds.z_end - search_span)
    {
        return false;
    }
    return true;
}

std::vector<FAtom> GridParallel::exchange_ghost_atoms(
    const std::vector<FAtom>& owned_atoms,
    const DomainBounds& bounds,
    const DomainDecomposition& decomp,
    MPI_Comm comm) const
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const int search_span = box_edge_length > 0.0
                                ? std::max(1, static_cast<int>(std::ceil(sradius / box_edge_length)))
                                : 1;
    const auto coord = rank_domain_coord(rank, decomp);

    std::vector<std::vector<FAtom>> send_atoms;
    std::vector<int> neighbors;
    for (int dz = -1; dz <= 1; dz++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                {
                    continue;
                }
                const int neighbor_rank = domain_rank_from_coord(coord[0] + dx,
                                                                 coord[1] + dy,
                                                                 coord[2] + dz,
                                                                 decomp);
                if (neighbor_rank == MPI_PROC_NULL)
                {
                    continue;
                }

                std::vector<FAtom> layer_atoms;
                for (const auto& atom : owned_atoms)
                {
                    if (atom_in_ghost_layer(atom, bounds, dx, dy, dz, search_span))
                    {
                        layer_atoms.push_back(atom);
                    }
                }
                neighbors.push_back(neighbor_rank);
                send_atoms.push_back(std::move(layer_atoms));
            }
        }
    }

    const int nneighbor = static_cast<int>(neighbors.size());
    std::vector<int> send_counts(nneighbor);
    std::vector<int> recv_counts(nneighbor);
    std::vector<MPI_Request> count_requests(nneighbor * 2);
    for (int i = 0; i < nneighbor; i++)
    {
        send_counts[i] = static_cast<int>(send_atoms[i].size());
        MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], 10, comm, &count_requests[i]);
        MPI_Isend(&send_counts[i], 1, MPI_INT, neighbors[i], 10, comm, &count_requests[nneighbor + i]);
    }
    if (!count_requests.empty())
    {
        MPI_Waitall(static_cast<int>(count_requests.size()), count_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<std::vector<double>> send_buffers(nneighbor);
    std::vector<std::vector<double>> recv_buffers(nneighbor);
    std::vector<MPI_Request> atom_requests(nneighbor * 2);
    for (int i = 0; i < nneighbor; i++)
    {
        pack_atoms(send_atoms[i], send_buffers[i]);
        recv_buffers[i].resize(recv_counts[i] * FATOM_PACK_SIZE);
        MPI_Irecv(recv_buffers[i].data(),
                  static_cast<int>(recv_buffers[i].size()),
                  MPI_DOUBLE,
                  neighbors[i],
                  11,
                  comm,
                  &atom_requests[i]);
        MPI_Isend(send_buffers[i].data(),
                  static_cast<int>(send_buffers[i].size()),
                  MPI_DOUBLE,
                  neighbors[i],
                  11,
                  comm,
                  &atom_requests[nneighbor + i]);
    }
    if (!atom_requests.empty())
    {
        MPI_Waitall(static_cast<int>(atom_requests.size()), atom_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<FAtom> ghost_atoms;
    for (const auto& buffer : recv_buffers)
    {
        std::vector<FAtom> received;
        unpack_atoms(buffer, received);
        ghost_atoms.insert(ghost_atoms.end(), received.begin(), received.end());
    }
    return ghost_atoms;
}

void GridParallel::rebuild_local_search_grid(const std::vector<FAtom>& owned_atoms,
                                             const std::vector<FAtom>& ghost_atoms)
{
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

    auto add_atom = [this](const FAtom& atom) {
        const int bx = atom_box_x(atom);
        const int by = atom_box_y(atom);
        const int bz = atom_box_z(atom);
        atoms_in_box[bx][by][bz].push_back(atom);
    };

    for (const auto& atom : owned_atoms)
    {
        add_atom(atom);
    }
    for (const auto& atom : ghost_atoms)
    {
        add_atom(atom);
    }
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
            this->Construct_Adjacent_near_box_local(atom);
        }
    }

    double t_end = omp_get_wtime();
    return t_end - t_start;
}

double GridParallel::Construct_Adjacent_parallel(const UnitCell& ucell, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double t_start = MPI_Wtime();

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

    all_adj_info.clear();
    all_adj_info.resize(ucell.ntype);
    for (int i = 0; i < ucell.ntype; i++)
    {
        all_adj_info[i].resize(ucell.atoms[i].na);
    }

    const DomainDecomposition decomp = choose_domain_decomposition(size);
    const DomainBounds bounds = rank_domain_bounds(rank, decomp);

    std::vector<FAtom> owned_search_atoms;
    for (const auto& atom : flat_atoms)
    {
        if (atom_in_domain(atom, bounds))
        {
            owned_search_atoms.push_back(atom);
        }
    }

    std::vector<FAtom> ghost_atoms = exchange_ghost_atoms(owned_search_atoms, bounds, decomp, comm);
    rebuild_local_search_grid(owned_search_atoms, ghost_atoms);

    std::vector<std::pair<int, int>> local_atoms;
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
            if (atom_in_domain(atom, bounds))
            {
                local_atoms.push_back({i_type, j_atom});
            }
        }
    }

    int my_count = static_cast<int>(local_atoms.size());

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int local_idx = 0; local_idx < my_count; local_idx++)
        {
            const int i_type = local_atoms[local_idx].first;
            const int j_atom = local_atoms[local_idx].second;

            FAtom atom(ucell.atoms[i_type].tau[j_atom].x,
                       ucell.atoms[i_type].tau[j_atom].y,
                       ucell.atoms[i_type].tau[j_atom].z,
                       i_type,
                       j_atom,
                       0, 0, 0);
            this->Construct_Adjacent_near_box_local(atom);
        }
    }

    int send_count = 0;
    std::vector<std::vector<NeighborEntry>> my_entries(my_count);
    std::vector<int> atom_id_buf(my_count * 2);
    for (int local_idx = 0; local_idx < my_count; local_idx++)
    {
        const int i_type = local_atoms[local_idx].first;
        const int j_atom = local_atoms[local_idx].second;
        atom_id_buf[local_idx * 2] = i_type;
        atom_id_buf[local_idx * 2 + 1] = j_atom;
        serialize_neighbors(i_type, j_atom, my_entries[local_idx]);
        send_count += static_cast<int>(my_entries[local_idx].size());
    }

    std::vector<int> size_buf(my_count);
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

    int* recv_double_counts = nullptr;
    int* recv_double_displs = nullptr;
    if (rank == 0)
    {
        recv_double_counts = new int[size];
        recv_double_displs = new int[size];
        for (int i = 0; i < size; i++)
        {
            recv_double_counts[i] = recv_counts[i] * 8;
            recv_double_displs[i] = recv_displs[i] * 8;
        }
    }

    MPI_Gatherv(data_buf, send_count * 8, MPI_DOUBLE,
                rank == 0 ? recv_buf : nullptr,
                rank == 0 ? recv_double_counts : nullptr,
                rank == 0 ? recv_double_displs : nullptr,
                MPI_DOUBLE, 0, comm);

    delete[] data_buf;
    if (rank == 0)
    {
        delete[] recv_double_counts;
        delete[] recv_double_displs;
    }

    int* size_recv_counts = nullptr;
    int* size_recv_displs = nullptr;
    int* size_recv_buf = nullptr;
    int* atom_id_recv_counts = nullptr;
    int* atom_id_recv_displs = nullptr;
    int* atom_id_recv_buf = nullptr;

    if (rank == 0)
    {
        size_recv_counts = new int[size];
        size_recv_displs = new int[size];
        atom_id_recv_counts = new int[size];
        atom_id_recv_displs = new int[size];
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
        int total_local_atoms = 0;
        for (int i = 0; i < size; i++)
        {
            total_local_atoms += size_recv_counts[i];
            atom_id_recv_counts[i] = size_recv_counts[i] * 2;
            atom_id_recv_displs[i] = size_recv_displs[i] * 2;
        }
        size_recv_buf = new int[total_local_atoms];
        atom_id_recv_buf = new int[total_local_atoms * 2];
    }

    MPI_Gatherv(size_buf.data(), my_count, MPI_INT,
                rank == 0 ? size_recv_buf : nullptr,
                rank == 0 ? size_recv_counts : nullptr,
                rank == 0 ? size_recv_displs : nullptr,
                MPI_INT, 0, comm);

    MPI_Gatherv(atom_id_buf.data(), my_count * 2, MPI_INT,
                rank == 0 ? atom_id_recv_buf : nullptr,
                rank == 0 ? atom_id_recv_counts : nullptr,
                rank == 0 ? atom_id_recv_displs : nullptr,
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
                    int entry_idx = total_idx * 8;
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

                const int atom_id_offset = (size_recv_displs[p] + i) * 2;
                const int i_type = atom_id_recv_buf[atom_id_offset];
                const int j_atom = atom_id_recv_buf[atom_id_offset + 1];
                deserialize_neighbors(i_type, j_atom, entries);
            }
        }

        delete[] recv_counts;
        delete[] recv_displs;
        delete[] recv_buf;
        delete[] size_recv_counts;
        delete[] size_recv_displs;
        delete[] size_recv_buf;
        delete[] atom_id_recv_counts;
        delete[] atom_id_recv_displs;
        delete[] atom_id_recv_buf;
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
