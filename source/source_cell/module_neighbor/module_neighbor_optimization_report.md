# ABACUS module_neighbor 近邻搜索优化报告

## 1. 修改范围

本次修改集中在旧近邻搜索模块：

```text
source/source_cell/module_neighbor/sltk_grid.h
source/source_cell/module_neighbor/sltk_grid.cpp
source/source_cell/module_neighbor/sltk_grid_parallel.*
```

其中真正进入当前主搜索路径的是 `Grid` / `Grid_Driver`。`GridParallel` 是上一阶段留下的 MPI 空间域划分候选实现，本报告也一并说明，但这次主要提交的是 `Grid` 主路径里的自适应剪枝和 OpenMP 调度优化。

整体思路是：不改近邻判定标准，只减少没必要检查的候选 box 和候选原子。最后是否加入邻接表，仍然由原来的距离条件决定：

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

## 5. 自适应搜索策略：密度感知网格 + box 包围盒剪枝

局部邻域搜索已经减少了 box 数量，但旧实现里 box 的划分仍然比较粗：`box_nx/y/z` 直接来自扩胞层数，一个周期镜像 cell 基本对应一个 box。这样在高密度体系里，一个 box 可能塞入较多原子，进入 box 之后仍要做很多逐原子距离判断。

这次把步骤一做成两层：

1. 构建网格时根据扩胞后的原子密度自适应选择 `box_edge_length`。
2. 搜索时继续使用每个 box 的真实原子包围盒 `box_bounds` 过滤不可能命中的候选 box。

最初没有用ai时，我们想到用类似蒙特卡洛方法，随机采样一些空间点统计空间密度，大概得到整体的空间密度分布。不过经过与ai讨论后，意识到如果引入随机数，网格划分会受随机种子影响，测试也会变得不稳定。因此这里没有在生产代码里使用真正随机的蒙特卡洛采样。当前实现用确定性的密度估计：扩胞后的原子总数除以坐标包围盒体积，然后反推出合适的 box 边长。这个做法比随机采样更便宜，也更容易复现。

### 5.1 密度感知网格划分

在 `setMemberVariables()` 里，先统计原胞原子数和扩胞层数：

```cpp
int natom_total = 0;
for (int i = 0; i < ucell.ntype; i++)
{
    natom_total += ucell.atoms[i].na;
}

const int image_nx = std::max(1, glayerX + glayerX_minus);
const int image_ny = std::max(1, glayerY + glayerY_minus);
const int image_nz = std::max(1, glayerZ + glayerZ_minus);
const double replicated_atoms = static_cast<double>(natom_total)
                              * static_cast<double>(image_nx)
                              * static_cast<double>(image_ny)
                              * static_cast<double>(image_nz);
```

再根据扩胞后坐标范围估计平均密度，并把目标设成每个 box 大约容纳 8 个原子：

```cpp
const double x_range = std::max(x_max - x_min, 1.0e-12);
const double y_range = std::max(y_max - y_min, 1.0e-12);
const double z_range = std::max(z_max - z_min, 1.0e-12);
const double volume = x_range * y_range * z_range;
const double target_atoms_per_box = 8.0;
const double safe_radius = std::max(sradius, 1.0e-8);
const double min_box_edge = safe_radius / 3.0;
const double max_box_edge = safe_radius + 0.1;

double adaptive_edge = max_box_edge;
if (replicated_atoms > 0.0 && volume > 0.0)
{
    const double density = replicated_atoms / volume;
    if (density > 0.0)
    {
        adaptive_edge = std::cbrt(target_atoms_per_box / density);
    }
}
```

最后把边长限制在 `[sradius / 3, sradius + 0.1]`。这样高密度时 box 会变小，但不会小到让邻域层数无限扩大；低密度时 box 会接近原来的半径尺度，避免创建大量空 box：

```cpp
this->box_edge_length = std::max(min_box_edge, std::min(max_box_edge, adaptive_edge));
const int max_boxes_per_axis = 512;
this->box_edge_length = std::max(this->box_edge_length, x_range / max_boxes_per_axis);
this->box_edge_length = std::max(this->box_edge_length, y_range / max_boxes_per_axis);
this->box_edge_length = std::max(this->box_edge_length, z_range / max_boxes_per_axis);

this->box_nx = std::max(1, static_cast<int>(std::ceil(x_range / box_edge_length)) + 1);
this->box_ny = std::max(1, static_cast<int>(std::ceil(y_range / box_edge_length)) + 1);
this->box_nz = std::max(1, static_cast<int>(std::ceil(z_range / box_edge_length)) + 1);
```

### 5.2 构建阶段同步维护 `box_bounds`

在 `sltk_grid.h` 中保留每个 box 的原子包围盒：

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
this->getBox(box_i_x, box_i_y, box_i_z, x, y, z);
box_i_x = std::max(0, std::min(this->box_nx - 1, box_i_x));
box_i_y = std::max(0, std::min(this->box_ny - 1, box_i_y));
box_i_z = std::max(0, std::min(this->box_nz - 1, box_i_z));
this->atoms_in_box[box_i_x][box_i_y][box_i_z].push_back(atom);
this->box_bounds[box_i_x][box_i_y][box_i_z].add_atom(atom);
```

这里的关键点是：原子不再按 `ix + glayerX_minus` 这种扩胞层号直接入箱，而是按真实坐标通过 `getBox()` 落到空间网格里。这样高密度区域会被拆到更多小 box，低密度区域则保持较粗的网格。

### 5.3 搜索阶段先判断 box 是否可能命中

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

### 5.4 剪枝判断的具体逻辑

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

这就是本次“自适应”的核心：网格尺寸先根据整体密度调节，搜索阶段再根据每个 box 内真实原子分布做精确剪枝。最终是否加入近邻表仍由 `dr <= sradius2` 判断，所以这不是近似搜索。

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

这次加入 `box_bounds` 后，MPI 局部网格重建也需要同步维护包围盒，否则局部搜索会拿不到候选 box 的边界信息。对应修正是让 `rebuild_local_search_grid()` 在重建 `atoms_in_box` 时同时清空、扩容并更新 `box_bounds`：

```cpp
atoms_in_box.clear();
box_bounds.clear();
atoms_in_box.resize(box_nx);
box_bounds.resize(box_nx);

auto add_atom = [this](const FAtom& atom) {
    const int bx = atom_box_x(atom);
    const int by = atom_box_y(atom);
    const int bz = atom_box_z(atom);
    atoms_in_box[bx][by][bz].push_back(atom);
    box_bounds[bx][by][bz].add_atom(atom);
};
```

这一块目前作为 MPI 候选实现保留，还没有替换所有生产路径；当前代码状态是并行局部网格和主路径使用同一套 `atoms_in_box` / `box_bounds` 数据约定。

## 8. 测试方式

为了只验证近邻搜索核心，本次使用已有独立 runner：

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

测试使用 4 线程：

```bash
OMP_NUM_THREADS=4 ./sltk_material_runtime_runner
```

测试判断两件事：

1. 正常版本和 `-DSLTK_DISABLE_ADAPTIVE_BOX_PRUNING` 版本的 `avg_neighbors` 一致，证明包围盒剪枝没有改变近邻结果。
2. `build_ms` 下降，证明近邻表构建加速。

## 9. 测试结果

### 9.1 4 线程密度感知网格

| 体系 | avg_neighbors | build_ms | total_ms |
| --- | ---: | ---: | ---: |
| Al 1000 | 12 | 2.7877 | 2.9389 |
| Si 2000 | 4 | 4.7504 | 4.9634 |
| NaCl 3000 | 6 | 6.0373 | 6.3823 |
| TiO2 4200 | 4 | 8.5563 | 9.0436 |

这组结果使用密度感知网格和 `box_bounds` 剪枝。`avg_neighbors` 分别为 Al 12、Si 4、NaCl 6、TiO2 4，符合这些测试晶体在当前截断半径下的近邻数预期。

### 9.2 关闭包围盒剪枝的对照

编译时加入：

```bash
-DSLTK_DISABLE_ADAPTIVE_BOX_PRUNING
```

| 体系 | avg_neighbors | build_ms | total_ms |
| --- | ---: | ---: | ---: |
| Al 1000 | 12 | 3.1769 | 3.3182 |
| Si 2000 | 4 | 4.9624 | 5.1994 |
| NaCl 3000 | 6 | 5.9916 | 6.3994 |
| TiO2 4200 | 4 | 9.4256 | 9.9215 |

两组 `avg_neighbors` 完全一致，说明 `box_bounds` 剪枝只减少候选 box 进入逐原子距离计算的次数，不影响最终近邻表。耗时上，开启剪枝后 Al、Si、TiO2 更快，NaCl 这组数据基本持平。

## 10. 总体总结

本项目的优化过程是从“减少无效计算”开始，再逐步叠加并行和负载均衡。旧版近邻搜索虽然已经有 box 数据结构，但实际构建邻接表时仍会产生大量不必要的候选遍历。我们首先把搜索范围限制到当前原子附近的局部 box，避免每个原子都扫描扩胞后的完整空间；随后在原子任务层面引入 OpenMP，使不同原子的近邻搜索可以并行执行。

在此基础上，本次进一步加入密度感知网格。具体做法是先用扩胞后的原子数和坐标包围盒估计平均密度，再把目标控制在每个 box 大约 8 个原子，由此确定 `box_edge_length` 和 `box_nx/y/z`。原子入箱时不再直接使用扩胞层号，而是通过 `getBox()` 按真实坐标进入空间网格。这样高密度区域会被拆成更多小 box，低密度区域则避免创建过细的空网格。

同时，代码仍然维护每个 box 内真实原子的坐标包围盒 `box_bounds`。搜索时先判断当前原子到候选 box 包围盒的最小可能距离，如果这个距离已经超过搜索半径，就直接跳过整个 box。这个判断发生在逐原子距离计算之前，因此能减少无效距离计算；最终是否加入近邻表仍由 `dr <= sradius2` 决定，所以不会改变近邻判定逻辑。

最后，考虑到不同原子附近候选数量不完全一致，我们将 OpenMP 调度方式从 `static` 改为 `guided,16`。这样线程不会固定处理一大段原子，而是在计算后期继续动态领取剩余任务，减少线程等待。测试结果表明，密度感知网格下开启或关闭包围盒剪枝时 `avg_neighbors` 完全一致；开启剪枝后多数测试体系的构建耗时进一步下降。整体来看，本次优化没有改变原有接口和近邻判定逻辑，但减少了近邻表构建阶段的候选扫描和距离判断开销。
