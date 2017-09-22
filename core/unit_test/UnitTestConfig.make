KOKKOS_PATH = ../..

# See $(KOKKOS_PATH)/Makefile.kokkos and $(KOKKOS_PATH)/generate_makefile.bash
KOKKOS_ARCH_OPTIONS="AMDAVX ARMv80 ARMv81 ARMv8-ThunderX Power7 Power8 Power9 \
	 WSM SNB HSW BDW SKX KNC KNL Kepler30 Kepler32 Kepler35 Kepler37 Maxwell50 \
	 Maxwell52 Maxwell53 Pascal60 Pascal61"
#KOKKOS_ARCH_OPTIONS="AMDAVX"

KOKKOS_DEVICE_OPTIONS="Cuda ROCm OpenMP Pthread Serial Qthreads"
#KOKKOS_DEVICE_OPTIONS="Cuda"

# Configure paths to enable environment query in Makefile.kokkos to work
ROCM_HCC_PATH="config"
CXX="./config/cxx"
ipath=env CXX=$(CXX) env PATH=./config:$$PATH env ROCM_HCC_PATH=$(ROCM_HCC_PATH)

# Defined in core/src/Makefile -- this should be consistent
KOKKOS_MAKEFILE=Makefile.kokkos
KOKKOS_CMAKEFILE=gen_kokkos.cmake

# Defined in Makefile.kokkos -- this should be consistent
KOKKOS_INTERNAL_CONFIG_TMP=KokkosCore_config.tmp
KOKKOS_CONFIG_HEADER=KokkosCore_config.h

# diff => 0 is no difference.  if => 0 is false
d='\#'
testmake=if grep $1 $2 | grep $3 2>&1 1> /dev/null; then echo OK $d $3 in $2; else echo not OK $d $3 in $2 \($1\); fi
# For the configure, need to do case insensitivity
testconf=if test `diff config/tmpstore/$1 config/results/$1 | grep define` !=''; then echo OK $d $1; else echo not OK $d $1; fi
#testconf=if diff config/tmpstore/$1 config/results/$1; then echo OK $d $1; else echo not OK $d $1; fi
#testconf=if test `diff config/tmpstore/$1 config/results/$1 | grep define` !=''; then echo OK $d $1; else echo not OK $d $1; fi

# testing tmp and cmakefile files is unnecessary
test:
	for karch in "$(KOKKOS_ARCH_OPTIONS)"; do \
	  for device in "$(KOKKOS_DEVICE_OPTIONS)"; do \
	     $(ipath) KOKKOS_DEVICES=$$device KOKKOS_ARCH=$$karch make -e -f ../src/Makefile build-makefile-cmake-kokkos; \
		 rm -f $(KOKKOS_INTERNAL_CONFIG_TMP) $(KOKKOS_CMAKEFILE); \
		 prfx="$$karch"_"$$device"_; \
		 newmake=config/tmpstore/"$$prfx"$(KOKKOS_MAKEFILE);  \
		 newconf="$$prfx"$(KOKKOS_CONFIG_HEADER); \
		 mv $(KOKKOS_MAKEFILE)  $$newmake; \
		 mv $(KOKKOS_CONFIG_HEADER) config/tmpstore/$$newconf; \
		 $(call testmake,KOKKOS_ARCH,$$newmake,$$karch); \
		 $(call testmake,KOKKOS_DEVICES,$$newmake,$$device); \
		 $(call testconf,$$newconf); \
	  done; \
	done
#   The mapping of ARCH to #define is very complicated
#       so is not tested.
#		 $(call testconf,KOKKOS_ARCH,$$newconf,$$karch); \
