#include "sltk_grid_parallel.h"

#include "source_base/global_function.h"
#include "source_base/timer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

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
    box_bounds.clear();
    atoms_in_box.resize(box_nx);
    box_bounds.resize(box_nx);
    for (int i = 0; i < box_nx; i++)
    {
        atoms_in_box[i].resize(box_ny);
        box_bounds[i].resize(box_ny);
        for (int j = 0; j < box_ny; j++)
        {
            atoms_in_box[i][j].resize(box_nz);
            box_bounds[i][j].resize(box_nz);
        }
    }

    auto add_atom = [this](const FAtom& atom) {
        const int bx = atom_box_x(atom);
        const int by = atom_box_y(atom);
        const int bz = atom_box_z(atom);
        atoms_in_box[bx][by][bz].push_back(atom);
        box_bounds[bx][by][bz].add_atom(atom);
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

GridParallel::DomainBounds GridParallel::rank_domain_bounds_balanced(
    int rank,
    const DomainDecomposition& decomp) const
{
    const int rx = rank % decomp.px;
    const int ry = (rank / decomp.px) % decomp.py;
    const int rz = rank / (decomp.px * decomp.py);

    // Compute workload per box-slice along x direction
    std::vector<int> x_workload(box_nx, 0);
    for (int bx = 0; bx < box_nx; bx++)
    {
        for (int by = 0; by < box_ny; by++)
        {
            for (int bz = 0; bz < box_nz; bz++)
            {
                x_workload[bx] += static_cast<int>(atoms_in_box[bx][by][bz].size());
            }
        }
    }

    // Cumulative sum
    std::vector<int> cumsum(box_nx + 1, 0);
    for (int i = 0; i < box_nx; i++)
    {
        cumsum[i + 1] = cumsum[i] + x_workload[i];
    }
    const int total_work = cumsum[box_nx];
    const int work_per_rank = std::max(1, total_work / decomp.px);

    // Find x-range boundaries by workload
    const auto find_split = [&](int target) {
        if (target <= 0)
        {
            return 0;
        }
        if (target >= total_work)
        {
            return box_nx;
        }
        for (int i = 0; i < box_nx; i++)
        {
            if (cumsum[i + 1] > target)
            {
                return i + 1;
            }
        }
        return box_nx;
    };

    const int x_begin = (rx == 0) ? 0 : find_split(rx * work_per_rank);
    const int x_end = (rx == decomp.px - 1) ? box_nx : find_split((rx + 1) * work_per_rank);

    // Ensure minimum domain width to avoid empty domains
    const int x_width = std::max(x_end - x_begin, 1);
    const int x_end_fixed = std::min(x_begin + x_width, box_nx);

    // Within x-range, split y-range by box count (uniform decomposition)
    const auto split_begin = [](int n, int p, int coord) {
        return coord * (n / p) + std::min(coord, n % p);
    };
    const auto split_end = [&](int n, int p, int coord) {
        return split_begin(n, p, coord + 1);
    };

    const int y_begin = split_begin(box_ny, decomp.py, ry);
    const int y_end = split_end(box_ny, decomp.py, ry);
    const int z_begin = split_begin(box_nz, decomp.pz, rz);
    const int z_end = split_end(box_nz, decomp.pz, rz);

    return {x_begin, x_end_fixed, y_begin, y_end, z_begin, z_end};
}

int GridParallel::estimate_atom_workload(const FAtom& atom) const
{
    int bx = 0;
    int by = 0;
    int bz = 0;
    getBox(bx, by, bz, atom.x, atom.y, atom.z);
    bx = std::max(0, std::min(box_nx - 1, bx));
    by = std::max(0, std::min(box_ny - 1, by));
    bz = std::max(0, std::min(box_nz - 1, bz));

    if (box_edge_length <= 0.0)
    {
        return 1;
    }

    const int search_span = std::max(1, static_cast<int>(std::ceil(sradius / box_edge_length)));
    const int x_begin = std::max(0, bx - search_span);
    const int x_end = std::min(box_nx - 1, bx + search_span);
    const int y_begin = std::max(0, by - search_span);
    const int y_end = std::min(box_ny - 1, by + search_span);
    const int z_begin = std::max(0, bz - search_span);
    const int z_end = std::min(box_nz - 1, bz + search_span);

    int count = 0;
    for (int ix = x_begin; ix <= x_end; ix++)
    {
        for (int iy = y_begin; iy <= y_end; iy++)
        {
            for (int iz = z_begin; iz <= z_end; iz++)
            {
                count += static_cast<int>(atoms_in_box[ix][iy][iz].size());
            }
        }
    }
    return count;
}

double GridParallel::Construct_Adjacent_serial(const UnitCell& ucell)
{
    double t_start = MPI_Wtime();

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

    double t_end = MPI_Wtime();
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
    const DomainBounds bounds = rank_domain_bounds_balanced(rank, decomp);

    std::vector<FAtom> owned_search_atoms;
    for (const auto& atom : flat_atoms)
    {
        if (atom_in_domain(atom, bounds))
        {
            owned_search_atoms.push_back(atom);
        }
    }

    std::vector<FAtom> ghost_atoms = exchange_ghost_atoms(owned_search_atoms, bounds, decomp, comm);

    std::vector<std::vector<std::vector<AtomMap>>> saved_atoms_in_box;
    std::vector<std::vector<std::vector<BoxBounds>>> saved_box_bounds;
    std::swap(saved_atoms_in_box, atoms_in_box);
    std::swap(saved_box_bounds, box_bounds);

    rebuild_local_search_grid(owned_search_atoms, ghost_atoms);

    struct LocalAtomTask
    {
        int i_type;
        int j_atom;
        int workload;
    };

    std::vector<LocalAtomTask> local_atoms;
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
                const int workload = estimate_atom_workload(atom);
                local_atoms.push_back({i_type, j_atom, workload});
            }
        }
    }

    // Sort by workload descending: heavy atoms first
    // Then schedule(static,1) interleaves them across threads
    std::sort(local_atoms.begin(), local_atoms.end(),
              [](const LocalAtomTask& a, const LocalAtomTask& b) {
                  return a.workload > b.workload;
              });

    int my_count = static_cast<int>(local_atoms.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static, 1)
#endif
        for (int local_idx = 0; local_idx < my_count; local_idx++)
        {
            const int i_type = local_atoms[local_idx].i_type;
            const int j_atom = local_atoms[local_idx].j_atom;

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
        const int i_type = local_atoms[local_idx].i_type;
        const int j_atom = local_atoms[local_idx].j_atom;
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
    MPI_Allreduce(&send_count, &total_entries, 1, MPI_INT, MPI_SUM, comm);

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

    int* recv_counts = new int[size];
    int* recv_displs = new int[size];
    double* recv_buf = nullptr;

    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    recv_displs[0] = 0;
    for (int i = 1; i < size; i++)
    {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    int* recv_double_counts = new int[size];
    int* recv_double_displs = new int[size];
    for (int i = 0; i < size; i++)
    {
        recv_double_counts[i] = recv_counts[i] * 8;
        recv_double_displs[i] = recv_displs[i] * 8;
    }

    recv_buf = new double[total_entries * 8];

    MPI_Allgatherv(data_buf, send_count * 8, MPI_DOUBLE,
                   recv_buf, recv_double_counts, recv_double_displs,
                   MPI_DOUBLE, comm);

    delete[] data_buf;
    delete[] recv_double_counts;
    delete[] recv_double_displs;

    int* size_recv_counts = new int[size];
    int* size_recv_displs = new int[size];
    int* atom_id_recv_counts = new int[size];
    int* atom_id_recv_displs = new int[size];

    MPI_Allgather(&my_count, 1, MPI_INT, size_recv_counts, 1, MPI_INT, comm);

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
    int* size_recv_buf = new int[total_local_atoms];
    int* atom_id_recv_buf = new int[total_local_atoms * 2];

    MPI_Allgatherv(size_buf.data(), my_count, MPI_INT,
                   size_recv_buf, size_recv_counts, size_recv_displs,
                   MPI_INT, comm);

    MPI_Allgatherv(atom_id_buf.data(), my_count * 2, MPI_INT,
                   atom_id_recv_buf, atom_id_recv_counts, atom_id_recv_displs,
                   MPI_INT, comm);

    // Restore full atoms_in_box for deserialization on all ranks
    std::swap(saved_atoms_in_box, atoms_in_box);
    std::swap(saved_box_bounds, box_bounds);

    // Build flat index map for O(1) lookup during deserialization
    // Map key: packed (type, natom, cell_x, cell_y, cell_z) -> FAtom* in atoms_in_box
    std::unordered_map<uint64_t, FAtom*> atom_map;
    for (int bx = 0; bx < box_nx; bx++)
    {
        for (int by = 0; by < box_ny; by++)
        {
            for (int bz = 0; bz < box_nz; bz++)
            {
                for (auto& atom : atoms_in_box[bx][by][bz])
                {
                    // Pack 5 ints into uint64: type(16) | natom(20) | cx(10) | cy(9) | cz(9)
                    uint64_t key = (static_cast<uint64_t>(atom.type & 0xFFFF) << 48)
                                 | (static_cast<uint64_t>(atom.natom & 0xFFFFF) << 28)
                                 | (static_cast<uint64_t>(atom.cell_x & 0x3FF) << 18)
                                 | (static_cast<uint64_t>(atom.cell_y & 0x1FF) << 9)
                                 | (static_cast<uint64_t>(atom.cell_z & 0x1FF));
                    atom_map[key] = &atom;
                }
            }
        }
    }

    // All ranks deserialize the complete result
    {
        int total_idx = 0;
        for (int p = 0; p < size; p++)
        {
            int count = size_recv_counts[p];
            for (int i = 0; i < count; i++)
            {
                int num_entries = size_recv_buf[size_recv_displs[p] + i];
                const int atom_id_offset = (size_recv_displs[p] + i) * 2;
                const int i_type = atom_id_recv_buf[atom_id_offset];
                const int j_atom = atom_id_recv_buf[atom_id_offset + 1];

                for (int e = 0; e < num_entries; e++)
                {
                    int entry_idx = (total_idx + e) * 8;
                    int ntype   = static_cast<int>(recv_buf[entry_idx]);
                    int nnatom  = static_cast<int>(recv_buf[entry_idx + 1]);
                    int ncell_x = static_cast<int>(recv_buf[entry_idx + 2]);
                    int ncell_y = static_cast<int>(recv_buf[entry_idx + 3]);
                    int ncell_z = static_cast<int>(recv_buf[entry_idx + 4]);

                    uint64_t key = (static_cast<uint64_t>(ntype & 0xFFFF) << 48)
                                 | (static_cast<uint64_t>(nnatom & 0xFFFFF) << 28)
                                 | (static_cast<uint64_t>(ncell_x & 0x3FF) << 18)
                                 | (static_cast<uint64_t>(ncell_y & 0x1FF) << 9)
                                 | (static_cast<uint64_t>(ncell_z & 0x1FF));

                    auto it = atom_map.find(key);
                    if (it != atom_map.end())
                    {
                        all_adj_info[i_type][j_atom].push_back(it->second);
                    }
                }
                total_idx += num_entries;
            }
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
