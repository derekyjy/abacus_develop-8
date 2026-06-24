remove_definitions(-D__MLALGO)
remove_definitions(-D__CUDA)
#include "module_/.h"
remove_definitions(-D__ROCM)
remove_definitions(-D__EXX)

AddTest(
  TARGET MODULE_CELL_NEIGHBOR_sltk_atom
  SOURCES sltk_atom_test.cpp ../sltk_atom.cpp
)

AddTest(
  TARGET MODULE_CELL_NEIGHBOR_sltk_grid
  LIBS parameter ${math_libs} base device cell_info 
  SOURCES sltk_grid_test.cpp ../sltk_grid.cpp ../sltk_atom.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_CELL_NEIGHBOR_sltk_atom_arrange
  LIBS parameter ${math_libs} base device cell_info 
  SOURCES sltk_atom_arrange_test.cpp  ../sltk_atom_arrange.cpp ../sltk_grid_driver.cpp ../sltk_grid.cpp
  ../sltk_atom.cpp
  ../../../source_io/module_output/output.cpp
)

if(ENABLE_MPI)
  add_executable(sltk_material_runtime_runner
    sltk_material_runtime_runner.cpp
    ../sltk_atom_arrange.cpp
    ../sltk_grid_driver.cpp
    ../sltk_grid.cpp
    ../sltk_atom.cpp
    ../sltk_grid_parallel.cpp
    ../../../source_io/module_output/output.cpp
  )
  target_include_directories(sltk_material_runtime_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/source
  )
  target_link_libraries(sltk_material_runtime_runner PRIVATE
    parameter ${math_libs} base device cell_info
  )
  set_target_properties(sltk_material_runtime_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
endif()