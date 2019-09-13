
# List of parallel device types 
HostArch=(SNB HSW SKX KNL)
DeviceArch=(Kepler35 Kepler37 Pascal60 Pascal61 Volta70)

MakeDevices=$1
CMakeDevices=$2

SRC_DIR=${KOKKOS_PATH}/core/unit_test/configuration/test-code

for harch in "${HostArch[@]}"
do
  harch_up=`echo $harch | tr a-z A-Z`
  CMAKE_HARCH="-DKokkos_ARCH_${harch_up}=ON"

  if [ ! -z "$DeviceArch" ]
  then
    for darch in "${DeviceArch[@]}"
    do
      darch_up=`echo $darch | tr a-z A-Z`
      CMAKE_DARCH="-DKokkos_ARCH_${darch_up}=ON"
      ${SRC_DIR}/test_config_options_list.bash "$MakeDevices" "$CMakeDevices" "$harch,$darch" "${CMAKE_HARCH} ${CMAKE_DARCH}"
    done
  else
    ${SRC_DIR}/test_config_options_list.bash "$MakeDevices" "$CMakeDevices" "$harch" "${CMAKE_HARCH}"
  fi 
done

