SCOTCH_PATH = /home/hcedwar/scotch/6.0.0
KOKKOS_PATH = ../..

vpath %.cpp ${KOKKOS_PATH}/example/ichol/src ${KOKKOS_PATH}/example/ichol/example 

EXAMPLE_HEADERS = $(wildcard $(KOKKOS_PATH)/example/ichol/src/*.hpp ${KOKKOS_PATH}/example/ichol/example/*.hpp )

default: build_all
	echo "End Build"
        
include $(KOKKOS_PATH)/Makefile.kokkos

ifeq ($(KOKKOS_INTERNAL_USE_CUDA), 1)
	CXX = $(NVCC_WRAPPER)
	CXXFLAGS ?= -O3
	LINK = $(CXX)
	LDFLAGS ?= -lpthread
else
	CXX ?= g++
	CXXFLAGS ?= -O3
	LINK ?= $(CXX)
	LDFLAGS ?= -lpthread
endif

KOKKOS_CXXFLAGS +=	\
	-I${KOKKOS_PATH}/example/ichol/src	\
	-I${KOKKOS_PATH}/example/ichol/example	\
	-I${SCOTCH_PATH}/include

EXE_EXAMPLE_ICHOL = KokkosExample_ichol
OBJ_EXAMPLE_ICHOL = example_chol_performance_device_pthread.o

TARGETS = $(EXE_EXAMPLE_ICHOL)

#TEST_TARGETS =

$(EXE_EXAMPLE_ICHOL) : $(OBJ_EXAMPLE_ICHOL) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) \
	$(OBJ_EXAMPLE_ICHOL) $(KOKKOS_LIBS) $(LIB) \
	-L${SCOTCH_PATH}/lib -lscotch  -lscotcherr  -lscotcherrexit \
	-o $(EXE_EXAMPLE_ICHOL)

build_all : $(TARGETS)

test : build_all

clean: kokkos-clean
	rm -f *.o $(TARGETS)

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(EXAMPLE_HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

