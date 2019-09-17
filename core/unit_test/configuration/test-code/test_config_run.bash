
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
make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX print-cxx-flags &> cxxflags

cd ../cmake
rm -rf *
cmake -DKokkos_SKIP_VALIDATION=ON \
      -DCMAKE_CXX_COMPILER=$CXX \
      $CMakeDevices \
      $CMakeArch \
      $CMakeOptions \
      $SRC_DIR &> config_out
cd ..
grep define gnu-make/KokkosCore_config.h | sort -u &> make_config_defines
grep define cmake/kokkos/KokkosCore_config.h | sort -u &> cmake_config_defines

diff make_config_defines cmake_config_defines &> config_defines_diff
diff_exists=`cat config_defines_diff | wc -l`
if [ $diff_exists -gt 0 ]
then
  echo ""
  echo ""
  echo "Failed #define test"
  echo Make: "make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h"
  echo CMake: "cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR"
  cat config_defines_diff
  echo "Sleeping for 3 seconds if you want to stop and explore..."
  echo ""
  sleep 3
else
  echo ""
  echo ""
  echo "Passed #define test"
  echo Make: "make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h"
  echo CMake: "cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR"
fi

#head multiple matches
#cut after generator expressions
#cut off trailing garbage
#sed change cmake list to spaces
#awk print each on new line
#grep remove empty lines
#sort and print unique flags
grep INTERFACE_COMPILE_OPTIONS cmake/kokkos/CMakeFiles/Export/lib/cmake/Kokkos/KokkosTargets.cmake \
  | head -n 1 \
  | cut -d":" -f3 \
  | cut -d">" -f 1 \
  | sed 's/;/ /g' \
  | awk -v RS=" " '{print}' \
  | grep -v -e '^$' \
  | sort | uniq > cmake_cxx_flags

#-I flags and -std= flags are not part of CMake's compile options
#that's fine, let's ignore thse below
#redunant lines - tail the last one
#awk print each on new line
#grep out blank lines
#grep out include flags
#grep out -std flags
#sort and print unique flags
tail -n 1 gnu-make/cxxflags \
  | awk -v RS=" " '{print}' \
  | grep -v -e '^$' \
  | grep -v '\-I' \
  | grep -v '\-std=' \
  | grep -v 'gcc-toolchain' \
  | sort | uniq > gnu_make_cxx_flags
diff gnu_make_cxx_flags cmake_cxx_flags &> config_cxxflags_diff
diff_exists=`cat config_cxxflags_diff | wc -l`

if [ $diff_exists -gt 0 ]
then
  echo ""
  echo ""
  echo "Failed CXXFLAGS test"
  echo Make: "make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h"
  echo CMake: "cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR"
  cat config_cxxflags_diff
  echo "Sleeping for 3 seconds if you want to stop and explore..."
  echo ""
  sleep 3
else
  echo ""
  echo ""
  echo "Passed CXXFLAGS test"
  echo Make: "make -f ${SRC_DIR}/Makefile KOKKOS_DEVICES=$MakeDevices KOKKOS_ARCH=$MakeArch $MakeOptions CXX=$CXX KokkosCore_config.h"
  echo CMake: "cmake -DCMAKE_CXX_COMPILER=$CXX $CMakeDevices $CMakeArch $CMakeOptions $SRC_DIR"
fi

