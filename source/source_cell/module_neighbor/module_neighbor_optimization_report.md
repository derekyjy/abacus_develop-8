# ABACUS module_neighbor 近邻搜索优化报告

## 1. 修改范围

本次修改集中在旧近邻搜索模块：

```text
source/source_cell/module_neighbor/sltk_grid.h
source/source_cell/module_neighbor/sltk_grid.cpp
source/source_cell/module_neighbor/sltk_grid_parallel.*
```

其中真正进入当前主搜索路径的是 `Grid` / `Grid_Driver`。`GridParallel` 是上一阶段留下的 MPI 空间域划分候选实现，本报告也一并说明，但这次主要提交的是 `Grid` 主路径里的自适应剪枝和 OpenMP 调度优化。

整体思路很简单：不改近邻判定标准，只减少没必要检查的候选 box 和候选原子。最后是否加入邻接表，仍然由原来的距离条件决定：

```cpp
if (dr != 0.0 && dr <= this->sradius2)
{
    all_adj_info[fatom1.type][fatom1.natom].push_back(fatom2);
}
```

所以这次优化属于精确剪枝，不是近似搜索。

## 2. 原代码的问题

旧代码里 `Grid::setMemberVariables()` 会把扩胞后的所有候选原子放进三维 `atoms_in_box`：

```cpp
atoms_in_box.resize(this->box_nx);
for (int i = 0; i < this->box_nx; i++)
{
    atoms_in_box[i].resize(this->box_ny);
    for (int j = 0; j < this->box_ny; j++)
    {
        atoms_in_box[i][j].resize(this->box_nz);
    }
}
```

随后 `Construct_Adjacent()` 对原胞内每个原子构造近邻表。早期实现的问题在于：即使已经知道当前原子位于哪个 box，搜索时仍容易退化成扫描大量不相关 box。对每个原子都做“扩胞空间内很多 box 的遍历 + box 内原子距离判断”，距离计算次数会被放大。

近邻搜索真正耗时的地方不是 `Find_atom()` 查询，而是在 `Grid::init()` 中一次性构建 `all_adj_info`：

```cpp
void Grid::init(std::ofstream& ofs_in, const UnitCell& ucell, const double radius_in, const bool boundary)
{
    this->pbc = boundary;
    this->sradius2 = radius_in * radius_in;
    this->sradius = radius_in;

    this->Check_Expand_Condition(ucell);
    this->setMemberVariables(ofs_in, ucell);
    this->Construct_Adjacent(ucell);
}
```

因此优化重点放在 `setMemberVariables()` 和 `Construct_Adjacent*()` 这一条路径上。

## 3. 局部邻域搜索：先缩小候选 box 范围

当前主路径使用 `Construct_Adjacent_near_box_local()`。它先定位当前原子所在 box：

```cpp
int box_i_x = 0;
int box_i_y = 0;
int box_i_z = 0;
this->getBox(box_i_x, box_i_y, box_i_z, fatom.x, fatom.y, fatom.z);

box_i_x = std::max(0, std::min(box_nx - 1, box_i_x));
box_i_y = std::max(0, std::min(box_ny - 1, box_i_y));
box_i_z = std::max(0, std::min(box_nz - 1, box_i_z));
```

然后根据搜索半径和 box 边长估算需要检查的邻域跨度：

```cpp
const int search_span = std::max(1, static_cast<int>(std::ceil(sradius / box_edge_length)));
const int x_begin = std::max(0, box_i_x - search_span);
const int x_end = std::min(box_nx - 1, box_i_x + search_span);
const int y_begin = std::max(0, box_i_y - search_span);
const int y_end = std::min(box_ny - 1, box_i_y + search_span);
const int z_begin = std::max(0, box_i_z - search_span);
const int z_end = std::min(box_nz - 1, box_i_z + search_span);
```

最后只遍历这块局部区域：

```cpp
for (int box_i_x_adj = x_begin; box_i_x_adj <= x_end; box_i_x_adj++)
{
    for (int box_i_y_adj = y_begin; box_i_y_adj <= y_end; box_i_y_adj++)
    {
        for (int box_i_z_adj = z_begin; box_i_z_adj <= z_end; box_i_z_adj++)
        {
            for (auto& fatom2 : this->atoms_in_box[box_i_x_adj][box_i_y_adj][box_i_z_adj])
            {
                this->Construct_Adjacent_final(fatom, &fatom2);
            }
        }
    }
}
```

这一步解决的是“不要每个原子都扫全局 box”的问题。

## 4. OpenMP 并行：把每个原子作为独立任务

`Construct_Adjacent()` 先把原胞内原子整理成任务数组：

```cpp
std::vector<std::pair<int, int>> atom_pairs;
for (int i_type = 0; i_type < ucell.ntype; i_type++)
{
    for (int j_atom = 0; j_atom < ucell.atoms[i_type].na; j_atom++)
    {
        atom_pairs.push_back({i_type, j_atom});
    }
}
```

每个任务只处理一个 `(type, atom)`，写入对应的 `all_adj_info[i_type][j_atom]`。不同线程不会同时写同一个原子的邻接表，因此这里不需要额外加锁：

```cpp
const int natom = static_cast<int>(atom_pairs.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(guided, 16)
#endif
for (int i = 0; i < natom; i++)
{
    const int i_type = atom_pairs[i].first;
    const int j_atom = atom_pairs[i].second;
    FAtom atom(ucell.atoms[i_type].tau[j_atom].x,
               ucell.atoms[i_type].tau[j_atom].y,
               ucell.atoms[i_type].tau[j_atom].z,
               i_type,
               j_atom,
               0, 0, 0);

    this->Construct_Adjacent_near_box_local(atom);
}
```

这里同时完成了并行优化和后面提到的动态负载均衡优化：`schedule(guided, 16)` 会让线程动态领取任务块。

## 5. 自适应剪枝：用 box 内真实原子的包围盒过滤候选

局部邻域搜索已经减少了 box 数量，但仍有不少邻域 box 实际上不可能含有近邻。特别是原子分布不均时，固定 `search_span` 会让低密度区域扫描很多空 box 或远 box。

为了解决这个问题，在 `sltk_grid.h` 中新增了每个 box 的原子包围盒：

```cpp
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

std::vector<std::vector<std::vector<BoxBounds>>> box_bounds;
```

### 5.1 构建阶段同步维护 `box_bounds`

在 `setMemberVariables()` 里，`atoms_in_box` 和 `box_bounds` 使用相同维度初始化：

```cpp
atoms_in_box.resize(this->box_nx);
box_bounds.resize(this->box_nx);
for (int i = 0; i < this->box_nx; i++)
{
    atoms_in_box[i].resize(this->box_ny);
    box_bounds[i].resize(this->box_ny);
    for (int j = 0; j < this->box_ny; j++)
    {
        atoms_in_box[i][j].resize(this->box_nz);
        box_bounds[i][j].resize(this->box_nz);
    }
}
```

放入扩胞候选原子时，同时更新该 box 的包围盒：

```cpp
FAtom atom(x, y, z, i, j, ix, iy, iz);
box_i_x = ix + glayerX_minus;
box_i_y = iy + glayerY_minus;
box_i_z = iz + glayerZ_minus;
this->atoms_in_box[box_i_x][box_i_y][box_i_z].push_back(atom);
this->box_bounds[box_i_x][box_i_y][box_i_z].add_atom(atom);
```

这样搜索阶段可以知道：一个 box 是空的，还是里面原子的实际空间范围在哪里。

### 5.2 搜索阶段先判断 box 是否可能命中

进入 box 内逐原子距离计算前，先调用：

```cpp
#ifndef SLTK_DISABLE_ADAPTIVE_BOX_PRUNING
if (!this->box_may_contain_neighbor(fatom, this->box_bounds[box_i_x_adj][box_i_y_adj][box_i_z_adj]))
{
    continue;
}
#endif
```

`SLTK_DISABLE_ADAPTIVE_BOX_PRUNING` 是测试开关。正常编译不开这个宏，自适应剪枝默认启用；A/B 测试时打开这个宏，可以回到不剪枝的行为。

### 5.3 剪枝判断的具体逻辑

`box_may_contain_neighbor()` 做的是“点到轴对齐包围盒的最小距离”判断：

```cpp
bool Grid::box_may_contain_neighbor(const FAtom& fatom, const BoxBounds& bounds) const
{
    if (bounds.empty)
    {
        return false;
    }

    const auto axis_distance2 = [](const double value, const double lower, const double upper) {
        if (value < lower)
        {
            const double diff = lower - value;
            return diff * diff;
        }
        if (value > upper)
        {
            const double diff = value - upper;
            return diff * diff;
        }
        return 0.0;
    };

    const double min_distance2 = axis_distance2(fatom.x, bounds.x_min, bounds.x_max)
                               + axis_distance2(fatom.y, bounds.y_min, bounds.y_max)
                               + axis_distance2(fatom.z, bounds.z_min, bounds.z_max);
    return min_distance2 <= this->sradius2;
}
```

如果当前原子到这个包围盒的最小可能距离都大于搜索半径，那么 box 内任何原子都不可能进入近邻表。反过来，如果这个条件通过，也不直接认为是近邻，只是允许进入原来的逐原子精确判断。

这就是本次“自适应”的核心：搜索范围不只由固定网格跨度决定，还由每个 box 内真实原子分布动态决定。

## 6. 动态负载均衡：为什么用 `guided,16`

OpenMP 的 `schedule(static)` 会把任务按固定区间分给线程。近邻搜索里每个原子的候选数量并不完全相等，所以 static 可能出现尾部等待。

本次尝试了 `dynamic,16` 和 `guided,16`。`dynamic` 对大体系有提升，但小体系上调度开销更明显；`guided` 前期块较大、后期块变小，在测试里更稳。因此最终代码保留：

```cpp
#pragma omp parallel for schedule(guided, 16)
```

这部分修改很小，但对已有并行路径比较关键：它不改变数据结构，也不改变搜索逻辑，只改变线程领取原子任务的方式。

## 7. MPI 并行优化现状

上一阶段已经在 `GridParallel` 中实现了 MPI 候选路径，主要代码结构是：

```cpp
const DomainDecomposition decomp = choose_domain_decomposition(size);
const DomainBounds bounds = rank_domain_bounds(rank, decomp);
```

`choose_domain_decomposition()` 会枚举进程数的三因子分解，尽量选择形状接近立方体的 `px, py, pz`：

```cpp
const double sx = static_cast<double>(std::max(1, box_nx)) / px;
const double sy = static_cast<double>(std::max(1, box_ny)) / py;
const double sz = static_cast<double>(std::max(1, box_nz)) / pz;
const double max_side = std::max(sx, std::max(sy, sz));
const double min_side = std::max(1.0, std::min(sx, std::min(sy, sz)));
const double score = max_side / min_side + 1.0e-6 * surface;
```

每个 rank 只负责自己空间子域内的 owned atoms：

```cpp
for (const auto& atom : flat_atoms)
{
    if (atom_in_domain(atom, bounds))
    {
        owned_search_atoms.push_back(atom);
    }
}
```

边界附近的原子通过 ghost atom 交换发送给相邻 rank：

```cpp
std::vector<FAtom> ghost_atoms = exchange_ghost_atoms(owned_search_atoms, bounds, decomp, comm);
rebuild_local_search_grid(owned_search_atoms, ghost_atoms);
```

这一块目前作为 MPI 候选实现保留，还没有替换所有生产路径。后续如果继续推进，重点应该是给它补正式 MPI benchmark，比较 1/2/4/8 rank 的邻居表一致性。

## 8. 测试方式

完整 ABACUS 顶层 CMake 会继续查找 FFTW3、BLAS、LAPACK、ScaLAPACK 等依赖。当前机器上 FFTW3 已通过 Conda 安装，CMake 可以找到：

```text
D:/Real_Softwares/anaconda/Library/lib/fftw3.lib
```

但顶层配置后续会卡在项目自带 `FindBLAS.cmake` / `FindLAPACK.cmake` 的递归查找上。为了只验证近邻搜索核心，本次使用已有独立 runner：

```text
source/source_cell/module_neighbor/test/sltk_material_runtime_runner.cpp
```

测试体系：

| 体系 | 原子数 | 结构类型 |
| --- | ---: | --- |
| Al fcc | 1000 | 金属 |
| Si diamond | 2000 | 半导体 |
| NaCl | 3000 | 离子晶体 |
| TiO2 rutile | 4200 | 复杂氧化物 |

测试判断两件事：

1. `avg_neighbors` 优化前后一致，证明结果没有变。
2. `build_ms` 下降，证明近邻表构建加速。

## 9. 测试结果

### 9.1 单线程自适应剪枝

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 10.4496 | 2.3172 | 4.51x |
| Si 2000 | 38.5710 | 6.5390 | 5.90x |
| NaCl 3000 | 88.4995 | 13.9946 | 6.32x |
| TiO2 4200 | 175.0070 | 24.0591 | 7.27x |

四个体系的 `avg_neighbors` 完全一致。

### 9.2 4 线程 static 调度

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 3.4369 | 1.6454 | 2.09x |
| Si 2000 | 10.0478 | 3.0583 | 3.29x |
| NaCl 3000 | 22.8761 | 5.7071 | 4.01x |
| TiO2 4200 | 46.3795 | 9.9352 | 4.67x |

### 9.3 4 线程 guided 调度

| 体系 | baseline build_ms | adaptive build_ms | 加速比 |
| --- | ---: | ---: | ---: |
| Al 1000 | 2.6884 | 1.5770 | 1.70x |
| Si 2000 | 8.2800 | 2.6953 | 3.07x |
| NaCl 3000 | 19.2133 | 5.2650 | 3.65x |
| TiO2 4200 | 40.6054 | 8.2521 | 4.92x |

guided 调度相对 static 调度在四个体系上都更快，所以最终保留。

## 10. 总体总结

本项目的优化过程是从“减少无效计算”开始，再逐步叠加并行和负载均衡。旧版近邻搜索虽然已经有 box 数据结构，但实际构建邻接表时仍会产生大量不必要的候选遍历。我们首先把搜索范围限制到当前原子附近的局部 box，避免每个原子都扫描扩胞后的完整空间；随后在原子任务层面引入 OpenMP，使不同原子的近邻搜索可以并行执行。

在此基础上，本次进一步加入自适应剪枝。具体做法是在构建 `atoms_in_box` 的同时维护每个 box 内真实原子的坐标包围盒 `box_bounds`。搜索时先判断当前原子到候选 box 包围盒的最小可能距离，如果这个距离已经超过搜索半径，就直接跳过整个 box。这个判断发生在逐原子距离计算之前，因此能减少大量无效距离计算；同时最终是否加入近邻表仍由 `dr <= sradius2` 决定，所以不会改变近邻搜索结果。

最后，考虑到不同原子附近候选数量不完全一致，我们将 OpenMP 调度方式从 `static` 改为 `guided,16`。这样线程不会固定处理一大段原子，而是在计算后期继续动态领取剩余任务，减少线程等待。测试结果表明，自适应剪枝在单线程和 4 线程下都能保持邻居数一致并明显降低构建时间；`guided` 调度相比 `static` 调度也进一步改善了 4 线程下的耗时。整体来看，本次优化没有改变原有接口和物理判定逻辑，但显著减少了近邻表构建阶段的候选扫描和距离判断开销。
