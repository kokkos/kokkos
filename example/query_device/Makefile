KOKKOS_PATH = ../..

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))

SRC = $(wildcard $(MAKEFILE_DIR)/*.cpp)
OBJ = $(SRC:$(MAKEFILE_DIR)/%.cpp=%.o)

default: build
	echo "Start Build"

include $(KOKKOS_PATH)/Makefile.kokkos

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = $(NVCC_WRAPPER)
CXXFLAGS = -O3
LINK = $(CXX)
LINKFLAGS = 
EXE = $(SRC:$(MAKEFILE_DIR)/%.cpp=%.cuda)
#KOKKOS_DEVICES = "Cuda,OpenMP"
#KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
CXXFLAGS = -O3
LINK = $(CXX)
LINKFLAGS =  
EXE = $(SRC:$(MAKEFILE_DIR)/%.cpp=%.host)
#KOKKOS_DEVICES = "OpenMP"
#KOKKOS_ARCH = "SNB"
endif

DEPFLAGS = -M

LIB =


build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.a *.o *.cuda *.host

# Compilation rules

%.o:$(MAKEFILE_DIR)/%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

