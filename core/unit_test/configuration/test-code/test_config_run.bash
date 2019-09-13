
SRC_DIR=${KOKKOS_PATH}/core/unit_test/configuration/test-code

# List of parallel device types 
MakeDevices=$1
CMakeDevices=$2
MakeArch=$3
CMakeArch=$4
MakeOptions=$5
CMakeOptions=$6

cd gnu-make
rm -rf *
make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h &>out

cd ../cmake
rm -rf *
cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR &> config_out
cd ..
grep define gnu-make/KokkosCore_config.h | sort -u &> make_config_defines
grep define cmake/kokkos/KokkosCore_config.h | sort -u &> cmake_config_defines

diff make_config_defines cmake_config_defines &> config_defines_diff
diff_exists=`cat config_defines_diff | wc -l`
if [ $diff_exists -gt 0 ]
then
  echo ""
  echo ""
  echo Make: "make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h"
  echo CMake: "cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR"
  cat config_defines_diff
fi

