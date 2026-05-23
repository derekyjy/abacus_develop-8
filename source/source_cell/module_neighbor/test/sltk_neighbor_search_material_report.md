# SLTK 旧近邻原子搜索算法测试总结报告

## 1. 测试目标

本报告只针对 ABACUS 旧近邻原子搜索实现：

```text
source/source_cell/module_neighbor/
```

核心测试对象为：

- `Grid`
- `Grid_Driver`
- `Grid_Driver::init()`
- `Grid_Driver::Find_atom()`

测试目标包括：

- 验证旧 SLTK 算法在不同材料体系和不同原子规模下能够构造有效近邻信息。
- 覆盖金属、半导体、离子晶体、复杂氧化物四类典型结构。
- 统计旧算法网格和近邻数据结构的构建时间。
- 统计旧算法近邻搜索核心数据结构的估算内存。
- 避免依赖完整 DFT 输入、赝势、轨道文件和 SCF 流程，只测试近邻搜索核心逻辑。

## 2. 测试内容

测试采用四类典型材料体系：

| 体系 | 原子数 | 基组场景 | 结构类型 | 测试目的 |
| --- | ---: | --- | --- | --- |
| Al fcc | 1,000 | PW | 金属 | 测试密堆积金属结构的近邻壳层 |
| Si diamond | 2,000 | LCAO | 半导体 | 测试四面体共价结构 |
| NaCl | 3,000 | PW | 离子晶体 | 测试 rock-salt 第一近邻配位 |
| TiO2 rutile | 4,200 | LCAO | 复杂氧化物 | 测试金红石 TiO2 的 Ti-O 配位关系 |

TiO2 使用金红石 rutile 常规胞构造。该常规胞包含 2 个 Ti 和 4 个 O，即 2 个 TiO2 化学式单元。为了保持完整周期结构和 Ti:O = 1:2 的化学计量，本测试采用 `10 x 10 x 7` 超胞，共 4,200 个原子，而不是截断到 4,000 个原子。4,000 不是 3 的倍数，无法构造严格化学计量的 TiO2 周期晶体。

## 3. 测试对象构造

旧算法 gtest 测试位于：

```text
source/source_cell/module_neighbor/test/sltk_atom_arrange_test.cpp
```

独立运行时间 runner 位于：

```text
source/source_cell/module_neighbor/test/sltk_material_runtime_runner.cpp
```

测试中定义材料描述结构：

```cpp
struct SltkMaterialCase
```

或在独立 runner 中定义：

```cpp
struct MaterialCase
```

主要字段包括：

- `name`：测试体系名称。
- `nx, ny, nz`：超胞在三个方向的复制次数。
- `lattice_x, lattice_y, lattice_z`：单个小晶胞在三个方向上的尺度。
- `cutoff`：近邻搜索半径。
- `ntype`：原子类型数量。
- `type_counts_per_cell`：每个小晶胞内各类型原子数量。
- `basis`：小晶胞内原子基矢位置。
- `expected_atoms`：目标原子数。

旧算法使用 ABACUS 原有 `UnitCell` 和 `Atom` 数据结构，因此测试通过函数：

```cpp
std::unique_ptr<UnitCell> make_sltk_crystal_case(const SltkMaterialCase& material)
```

或独立 runner 中的：

```cpp
std::unique_ptr<UnitCell> make_crystal_case(const MaterialCase& material)
```

构造旧算法需要的 `UnitCell`。

构造步骤如下：

1. 创建 `UnitCell` 对象。
2. 设置 `lat0`、`latvec`、`a1/a2/a3`、`omega`。
3. 设置原子类型数 `ntype` 和目标总原子数 `nat`。
4. 分配 `ucell->atoms = new Atom[ntype]`。
5. 按材料基元在超胞中生成原子坐标。
6. 按类型写入 `ucell->atoms[type].tau`。
7. 设置每种类型的原子数 `ucell->atoms[type].na`。

## 4. 测试代码思路

旧 SLTK 算法测试的核心流程为：

```cpp
Grid_Driver grid_d(PARAM.input.test_deconstructor, PARAM.input.test_grid);
grid_d.init(ofs, *ucell, material.cutoff, true);
grid_d.Find_atom(*ucell, type, atom, &adjs);
```

测试逻辑如下：

1. 构造旧算法使用的 `UnitCell`。
2. 调用 `Grid_Driver::init()` 构造旧算法近邻搜索数据。
3. 对每个原子调用 `Grid_Driver::Find_atom()` 查询近邻。
4. 检查 `AdjacentAtomInfo` 内部数组长度是否一致。
5. 检查每个原子至少有一个近邻。
6. 检查旧接口约定：自身原子被追加在返回列表末尾。
7. 统计平均近邻数、网格构建耗时和近邻搜索数据结构内存。

## 5. 计时与内存定义

### 5.1 网格构建时间

计时代码为：

```cpp
const auto build_start = std::chrono::steady_clock::now();
grid_d.init(ofs, *ucell, material.cutoff, true);
const auto build_finish = std::chrono::steady_clock::now();
const double build_ms =
    std::chrono::duration<double, std::milli>(build_finish - build_start).count();
```

`build_ms` 只统计 `Grid_Driver::init()` 的耗时，也就是旧 SLTK 算法中网格和近邻数据结构的构建时间。该阶段完成周期扩胞、候选原子生成、距离判断和邻接关系构造，是旧算法的主要耗时阶段。

### 5.2 内存统计

内存统计函数为：

```cpp
double estimate_grid_memory_mb(const Grid_Driver& grid)
```

统计口径是旧 SLTK 近邻搜索核心数据结构的估算内存，主要包括：

- `atoms_in_box`：按空间 box 保存的扩胞候选原子。
- `all_adj_info`：每个原子的近邻指针列表。

该指标不是整个进程的 RSS，也不是任务管理器看到的总内存，而是近邻搜索算法数据结构本身的估算内存。后续如果与 MPI 并行实现对比，建议每个 rank 使用同一口径统计本 rank 的近邻搜索数据结构内存，然后报告：

- `max_rank_memory_mb`：单个 rank 的最大内存压力。
- `sum_rank_memory_mb`：整个并行作业的总内存消耗。

## 6. 核心函数说明

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

主要字段包括：

- `adj_num`：近邻数量，不包含最后追加的自身原子。
- `ntype`：近邻原子类型。
- `natom`：近邻原子在该类型内的编号。
- `adjacent_tau`：近邻原子坐标。
- `box`：近邻原子所在的周期镜像 box。

旧接口有一个特殊约定：`Find_atom()` 会把自身原子追加在返回列表最后，因此测试中专门检查最后一个元素是否为自身。

## 7. 测试结果

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

平均近邻数只作为结果统计输出，不作为强制通过条件。从结果可以看到，随着原子数增加，旧 SLTK 算法构造近邻表耗时明显上升，近邻搜索数据结构内存也随体系规模增大而增加。

## 8. 结论

本次测试完成了对旧 SLTK 近邻原子搜索算法的材料覆盖验证。旧算法能够在四类典型材料体系上正确构造近邻信息，并保持旧接口 `AdjacentAtomInfo` 的行为约定。

从运行时间结果看，随着体系规模从 1,000 原子增加到 4,200 原子，网格构建时间从约 141 ms 增加到约 2555 ms，说明旧算法在大体系下存在明显性能压力。

从内存结果看，近邻搜索核心数据结构内存从约 1.41 MB 增加到约 10.39 MB。该内存主要由扩胞候选原子 `atoms_in_box` 和邻接指针列表 `all_adj_info` 构成。后续与并行实现比较时，应使用相同口径统计每个 rank 的近邻搜索数据结构内存，并同时报告最大 rank 内存和总内存。
