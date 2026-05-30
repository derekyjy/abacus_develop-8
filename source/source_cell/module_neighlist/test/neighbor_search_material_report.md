# 近邻原子搜索算法测试总结报告

## 1. 测试目标

本次测试面向 ABACUS 中近邻原子搜索相关的两套实现：

- 旧实现：`source/source_cell/module_neighbor/`，主要类为 `Grid`、`Grid_Driver`，也可称为 SLTK 旧算法。
- 新实现：`source/source_cell/module_neighlist/`，主要类为 `NeighborSearch`、`BinManager`、`NeighborList`。

测试目标包括：

- 验证两套算法在不同材料体系、不同原子规模下都能构造有效近邻表。
- 覆盖金属、半导体、离子晶体、复杂氧化物等典型结构类型。
- 对旧 SLTK 算法额外记录运行时间，用于后续和新算法做性能对比。
- 避免依赖完整 DFT 输入、赝势、轨道文件和 SCF 流程，只直接测试近邻搜索核心逻辑。

## 2. 测试内容

测试采用四类典型材料体系：

| 体系 | 原子数 | 基组场景 | 结构类型 | 测试目的 |
| --- | ---: | --- | --- | --- |
| Al fcc | 1,000 | PW | 金属 | 测试密堆积金属结构的近邻壳层 |
| Si diamond | 2,000 | LCAO | 半导体 | 测试四面体共价结构 |
| NaCl | 3,000 | PW | 离子晶体 | 测试 rock-salt 第一近邻配位 |
| TiO2 rutile | 4,200 | LCAO | 复杂氧化物 | 测试金红石 TiO2 的 Ti-O 配位关系 |

其中 TiO2 使用金红石 rutile 常规胞构造。该常规胞包含 2 个 Ti 和 4 个 O，即 2 个 TiO2 化学式单元。为了保持完整周期结构和 Ti:O = 1:2 的化学计量，本测试采用 10 x 10 x 7 超胞，共 4,200 个原子，而不是将结构截断到 4,000 个原子。4,000 不是 3 的倍数，无法构造严格化学计量的 TiO2 周期晶体。

## 3. 测试对象如何构建

### 3.1 新算法测试体系构造

新算法测试位于：

`source/source_cell/module_neighlist/test/neighbor_search_test.cpp`

核心测试为：

```cpp
TEST(NeighborSearchMaterialCoverage, BenchmarkSystemsFromProjectReport)
```

测试中定义了一个材料描述结构：

```cpp
struct MaterialNeighborCase
```

它包含以下信息：

- `name`：测试体系名称。
- `nx, ny, nz`：超胞在三个方向的复制次数。
- `lattice_x, lattice_y, lattice_z`：单个小晶胞在三个方向上的尺度。Al、Si、NaCl 使用三向相同尺度；TiO2 rutile 使用四方晶系尺度，满足 `a = b != c`。
- `cutoff`：近邻搜索半径。
- `ntype`：原子类型数量。
- `type_counts_per_cell`：每个小晶胞内各类型原子数量。
- `basis`：小晶胞内原子基矢位置。
- `expected_atoms`：目标原子数。

测试通过函数：

```cpp
UnitCellPlus make_crystal_case(const MaterialNeighborCase& material)
```

构造 `UnitCellPlus`。`UnitCellPlus` 是新 `module_neighlist` 测试用的轻量晶胞对象，实现了 `IAtomProvider` 接口，可以被 `NeighborSearch` 直接读取。

构造步骤如下：

1. 设置 `lat0 = 1.0`，简化单位换算。
2. 根据 `nx, ny, nz` 和 `lattice_x, lattice_y, lattice_z` 生成正交超胞晶格矢量。
3. 设置晶胞体积 `omega`。
4. 按 `basis` 在每个复制晶胞中生成原子坐标。
5. 按原子类型将坐标放入 `ucell.tau`。
6. 设置 `ucell.nat`、`ucell.na`，并调用 `compute_naa()` 生成类型累计索引。

### 3.2 旧 SLTK 算法测试体系构造

旧算法 gtest 测试位于：

`source/source_cell/module_neighbor/test/sltk_atom_arrange_test.cpp`

核心测试为：

```cpp
TEST_F(SltkAtomArrangeTest, MaterialCoverageAndRuntime)
```

测试中定义了：

```cpp
struct SltkMaterialCase
```

其字段与新算法测试基本一致。旧算法使用 ABACUS 原有 `UnitCell` 和 `Atom` 数据结构，因此测试通过函数：

```cpp
std::unique_ptr<UnitCell> make_sltk_crystal_case(const SltkMaterialCase& material)
```

构造旧算法需要的 `UnitCell`。

构造步骤如下：

1. 创建 `UnitCell` 对象。
2. 设置 `lat0`、`latvec`、`a1/a2/a3`、`omega`。
3. 设置原子类型数 `ntype` 和目标总原子数 `nat`。
4. 分配 `ucell->atoms = new Atom[ntype]`。
5. 按材料基元生成原子坐标。
6. 按类型写入 `ucell->atoms[type].tau`。
7. 设置每种类型的原子数 `ucell->atoms[type].na`。

### 3.3 独立 SLTK runner

由于 ABACUS 顶层 CMake 会额外查找 FFTW3、BLAS、LAPACK、ScaLAPACK 等依赖，而本次只需要测试旧近邻搜索算法，因此额外编写了一个独立 runner：

`source/source_cell/module_neighbor/test/sltk_material_runtime_runner.cpp`

该 runner 只链接旧近邻搜索所需的最小代码路径，并补充少量最小 stub，用于绕过完整 ABACUS 可执行程序依赖。它的目标是快速得到旧算法在四个材料体系上的运行时间。

## 4. 测试代码编写思路

### 4.1 新算法测试思路

新算法测试核心流程：

```cpp
NeighborSearch ns;
ns.init(ucell, material.cutoff, 0);
ns.build_neighbors();
NeighborList& list = ns.get_neighbor_list();
```

测试逻辑分为三层：

1. **输入构造正确性检查**
   - 检查 `type_counts_per_cell` 是否和 `ntype` 对应。
   - 检查 `basis` 中各类型原子数量是否符合预期。
   - 检查最终生成的 `ucell.nat` 是否等于目标原子数。

2. **执行近邻搜索**
   - `NeighborSearch::init()` 负责初始化搜索半径、周期扩胞、生成 `all_atoms`，并划分 `inside_atoms` 和 `ghost_atoms`。
   - `NeighborSearch::build_neighbors()` 调用 `BinManager` 建立 bin，并生成 `NeighborList`。

3. **输出近邻表检查**
   - 检查 `inside_atoms.size()` 是否等于目标原子数。
   - 检查 `NeighborList::nlocal` 是否等于目标原子数。
   - 检查 `numneigh` 和 `firstneigh` 的大小是否正确。
   - 检查每个原子至少有一个近邻。
   - 检查近邻列表中没有把自身原子记录为自己的近邻。

### 4.2 旧 SLTK gtest 测试思路

旧算法测试核心流程：

```cpp
Grid_Driver grid_d(PARAM.input.test_deconstructor, PARAM.input.test_grid);
grid_d.init(ofs, *ucell, material.cutoff, true);
grid_d.Find_atom(*ucell, type, atom, &adjs);
```

测试逻辑为：

1. 构造旧算法使用的 `UnitCell`。
2. 调用 `Grid_Driver::init()` 构造旧算法近邻数据。
3. 对每个原子调用 `Grid_Driver::Find_atom()` 查询近邻。
4. 检查 `AdjacentAtomInfo` 内部数组长度是否一致。
5. 检查每个原子至少有一个近邻。
6. 检查旧接口约定：自身原子被追加在返回列表末尾。
7. 记录 `Grid_Driver::init()` 的 wall-clock 时间。

计时代码使用：

```cpp
const auto start = std::chrono::steady_clock::now();
grid_d.init(ofs, *ucell, material.cutoff, true);
const auto finish = std::chrono::steady_clock::now();
```

并输出：

```text
[sltk] ... avg_neighbors=... build_ms=... memory_mb=...
```

### 4.3 独立 runner 编写思路

独立 runner 的代码思路与旧 SLTK gtest 基本一致，但不依赖 gtest 和顶层 CMake。

它直接执行：

```cpp
Grid_Driver grid_d;
grid_d.init(ofs, *ucell, material.cutoff, true);
```

然后遍历所有原子：

```cpp
grid_d.Find_atom(*ucell, type, atom, &adjs);
```

统计：

- 总近邻数 `total_neighbors`
- 有近邻的原子数 `atoms_with_neighbors`
- 平均近邻数 `average_neighbors`
- 网格构建耗时 `build_ms`
- 近邻搜索数据结构估算内存 `memory_mb`

如果出现原子数不对、近邻覆盖不足、自身原子不在末尾等情况，runner 会抛出异常并终止。

## 5. 核心函数说明

### 5.1 新算法核心函数

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `NeighborSearch::init()` | `module_neighlist/neighbor_search.cpp` | 初始化搜索半径、扩胞、inside/ghost 原子和近邻表 |
| `NeighborSearch::Check_Expand_Condition()` | `module_neighlist/neighbor_search.cpp` | 根据 cutoff 和晶格矢量计算周期扩胞层数 |
| `NeighborSearch::setMemberVariables()` | `module_neighlist/neighbor_search.cpp` | 生成周期镜像原子 `all_atoms` |
| `NeighborSearch::build_neighbors()` | `module_neighlist/neighbor_search.cpp` | 调用 `BinManager` 构造近邻表 |
| `BinManager::init_bins()` | `module_neighlist/bin_manager.cpp` | 初始化空间 bin |
| `BinManager::do_binning()` | `module_neighlist/bin_manager.cpp` | 将 inside/ghost 原子放入 bin |
| `BinManager::build_atom_neighbors()` | `module_neighlist/bin_manager.cpp` | 遍历相邻 27 个 bin，生成近邻列表 |

新算法最终输出：

```cpp
NeighborList
```

主要字段：

- `nlocal`：本地原子数量。
- `numneigh[i]`：第 `i` 个本地原子的近邻数量。
- `firstneigh[i]`：第 `i` 个本地原子的近邻 id 数组。

### 5.2 旧 SLTK 算法核心函数

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `Grid_Driver::init()` | 继承自 `Grid::init()` | 构造旧算法近邻搜索数据 |
| `Grid::Check_Expand_Condition()` | `module_neighbor/sltk_grid.cpp` | 根据 cutoff 计算周期扩胞层数 |
| `Grid::setMemberVariables()` | `module_neighbor/sltk_grid.cpp` | 生成扩胞后的候选原子 |
| `Grid::Construct_Adjacent()` | `module_neighbor/sltk_grid.cpp` | 遍历原胞原子并构造近邻 |
| `Grid::Construct_Adjacent_final()` | `module_neighbor/sltk_grid.cpp` | 计算距离并判断是否加入近邻列表 |
| `Grid_Driver::Find_atom()` | `module_neighbor/sltk_grid_driver.cpp` | 查询某个原子的近邻信息 |

旧算法最终输出：

```cpp
AdjacentAtomInfo
```

主要字段：

- `adj_num`：近邻数量，不包含最后追加的自身原子。
- `ntype`：近邻原子类型。
- `natom`：近邻原子在该类型内的编号。
- `adjacent_tau`：近邻原子坐标。
- `box`：近邻原子所在周期镜像 box。

旧接口有一个特殊约定：`Find_atom()` 会把自身原子追加在返回列表最后，因此测试中专门检查最后一个元素是否为自身。

## 6. 测试结果

### 6.1 新算法测试结果

独立 smoke run 直接链接 `neighbor_search.cpp` 和 `bin_manager.cpp`，对四个材料体系执行新算法近邻搜索：

| 体系 | 原子数 | 平均近邻数 | 结果 |
| --- | ---: | ---: | --- |
| Al fcc | 1,000 | 12.0 | 通过 |
| Si diamond | 2,000 | 4.0 | 通过 |
| NaCl | 3,000 | 6.0 | 通过 |
| TiO2 rutile | 4,200 | 4.0 | 通过 |

这些平均近邻数与构造的测试结构相符：

- fcc 金属结构平均近邻数约为 12。
- diamond 结构平均近邻数约为 4。
- NaCl rock-salt 结构平均近邻数约为 6。
- TiO2 rutile 中 Ti 的近邻 O 配位数为 6，O 的近邻 Ti 配位数为 3；按 Ti:O = 1:2 对所有原子取平均，平均近邻数为 `(1*6 + 2*3) / 3 = 4`。

### 6.2 旧 SLTK 算法运行时间测试结果

独立 SLTK runner 输出如下：

```text
[sltk] Al fcc / 1000 atoms / PW / metal: atoms=1000, avg_neighbors=12, build_ms=140.929, memory_mb=1.41178
[sltk] Si diamond / 2000 atoms / LCAO / semiconductor: atoms=2000, avg_neighbors=4, build_ms=573.257, memory_mb=2.63926
[sltk] NaCl / 3000 atoms / PW / ionic crystal: atoms=3000, avg_neighbors=6, build_ms=1343.78, memory_mb=5.31549
[sltk] TiO2 rutile / 4200 atoms / LCAO / complex oxide: atoms=4200, avg_neighbors=4, build_ms=2554.68, memory_mb=10.3932
```

整理为表格：

| 体系 | 原子数 | 平均近邻数 | 网格构建时间 ms | 内存 MB | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| Al fcc | 1,000 | 12.0 | 140.929 | 1.41178 | 通过 |
| Si diamond | 2,000 | 4.0 | 573.257 | 2.63926 | 通过 |
| NaCl | 3,000 | 6.0 | 1343.78 | 5.31549 | 通过 |
| TiO2 rutile | 4,200 | 4.0 | 2554.68 | 10.3932 | 通过 |

其中 `build_ms` 只统计 `Grid_Driver::init()`，即旧 SLTK 网格和近邻数据结构的构建时间。`memory_mb` 统计的是 `Grid_Driver::init()` 后近邻搜索核心数据结构的估算内存，主要包括 `atoms_in_box` 中保存的扩胞候选原子，以及 `all_adj_info` 中保存的邻接指针列表。平均近邻数只作为结果统计输出，不再作为强制通过条件。

如果后续与 MPI 并行实现对比，建议采用同一统计口径：每个 rank 分别统计本 rank 的近邻搜索数据结构内存，然后在报告中给出 `max_rank_memory_mb` 和 `sum_rank_memory_mb`。`max_rank_memory_mb` 反映单个进程的峰值压力，`sum_rank_memory_mb` 反映整个并行作业的总内存消耗。

从结果可以看到，随着原子数增加，旧 SLTK 算法构造近邻表耗时明显上升。其原因是旧算法虽然有 `atoms_in_box` 的数据结构，但实际构造近邻时仍然会在扩胞候选范围内做较大范围的遍历和距离判断。

## 7. 结论

本次测试完成了对新旧两套近邻原子搜索算法的材料覆盖验证。

新算法 `module_neighlist` 能在单 rank 情况下正确完成四类材料体系的近邻表构造，输出的平均近邻数符合预期。它使用 bin/cell list 思路，只检查当前 bin 及周围 bin 中的候选原子，因此更适合后续大规模体系优化。

旧算法 `module_neighbor` 的 SLTK 路径也能正确构造近邻信息，并保持旧接口 `AdjacentAtomInfo` 的行为约定。但从运行时间结果看，随着体系规模从 1,000 原子增加到 4,200 原子，构建时间从约 142 ms 增加到约 2568 ms，说明旧算法在大体系下存在明显性能压力。

当前测试主要验证单 rank 逻辑。新算法中虽然已有 `inside_atoms`、`ghost_atoms` 和 `decompose()` 等并行设计接口，但 `NeighborSearch::init()` 中 `mpi_size` 仍固定为 1，因此本测试尚未覆盖真实多 MPI rank 的区域划分和通信过程。
