KOKKOS_PATH=../../
build:
	mkdir -p 01_hello_world
	cd ./01_hello_world; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/01_hello_world/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 01_hello_world_lambda
	cd ./01_hello_world_lambda; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/01_hello_world_lambda/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 02_simple_reduce
	cd ./02_simple_reduce; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 02_simple_reduce_lambda
	cd ./02_simple_reduce_lambda; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce_lambda/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 03_simple_view
	cd ./03_simple_view; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/03_simple_view/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 03_simple_view_lambda
	cd ./03_simple_view_lambda; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/03_simple_view_lambda/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 04_simple_memoryspaces
	cd ./04_simple_memoryspaces; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/04_simple_memoryspaces/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 05_simple_atomics
	cd ./05_simple_atomics; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/05_simple_atomics/Makefile ${KOKKOS_OPTIONS}
	mkdir -p Advanced_Views
	cd ./Advanced_Views; \
	make build -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	mkdir -p Algorithms
	cd ./Algorithms; \
	make build -f ${KOKKOS_PATH}/example/tutorial/Algorithms/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	mkdir -p Hierarchical_Parallelism
	cd ./Hierarchical_Parallelism; \
	make build -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'

build-insource:
	cd ./01_hello_world; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./01_hello_world_lambda; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce_lambda; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./03_simple_view; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./03_simple_view_lambda; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./04_simple_memoryspaces; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./05_simple_atomics; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./Advanced_Views; \
	make build KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Algorithms; \
	make build KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Hierarchical_Parallelism; \
	make build KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
test:
	cd ./01_hello_world; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/01_hello_world/Makefile ${KOKKOS_OPTIONS}
	cd ./01_hello_world_lambda; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/01_hello_world_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce/Makefile ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce_lambda; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./03_simple_view; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/03_simple_view/Makefile ${KOKKOS_OPTIONS}
	cd ./03_simple_view_lambda; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/03_simple_view_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./04_simple_memoryspaces; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/04_simple_memoryspaces/Makefile ${KOKKOS_OPTIONS}
	cd ./05_simple_atomics; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/05_simple_atomics/Makefile ${KOKKOS_OPTIONS}
	cd ./Advanced_Views; \
	make test -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Algorithms; \
	make test -f ${KOKKOS_PATH}/example/tutorial/Algorithms/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Hierarchical_Parallelism; \
	make test -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'

test-insource:
	cd ./01_hello_world; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./01_hello_world_lambda; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce_lambda; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./03_simple_view; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./03_simple_view_lambda; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./04_simple_memoryspaces; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./05_simple_atomics; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./Advanced_Views; \
	make test KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Algorithms; \
	make test KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Hierarchical_Parallelism; \
	make test KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
clean:
	cd ./01_hello_world; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/01_hello_world/Makefile ${KOKKOS_OPTIONS}
	cd ./01_hello_world_lambda; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/01_hello_world_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce/Makefile ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce_lambda; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/02_simple_reduce_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./03_simple_view; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/03_simple_view/Makefile ${KOKKOS_OPTIONS}
	cd ./03_simple_view_lambda; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/03_simple_view_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./04_simple_memoryspaces; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/04_simple_memoryspaces/Makefile ${KOKKOS_OPTIONS}
	cd ./05_simple_atomics; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/05_simple_atomics/Makefile ${KOKKOS_OPTIONS}
	cd ./Advanced_Views; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Algorithms; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Algorithms/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Hierarchical_Parallelism; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/Makefile KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'

clean-insource:
	cd ./01_hello_world; \
	make clean ${KOKKOS_OPTIONS}
	cd ./01_hello_world_lambda; \
	make clean ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce; \
	make clean ${KOKKOS_OPTIONS}
	cd ./02_simple_reduce_lambda; \
	make clean ${KOKKOS_OPTIONS}
	cd ./03_simple_view; \
	make clean ${KOKKOS_OPTIONS}
	cd ./03_simple_view_lambda; \
	make clean ${KOKKOS_OPTIONS}
	cd ./04_simple_memoryspaces; \
	make clean ${KOKKOS_OPTIONS}
	cd ./05_simple_atomics; \
	make clean ${KOKKOS_OPTIONS}
	cd ./Advanced_Views; \
	make clean KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Algorithms; \
	make clean KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
	cd ./Hierarchical_Parallelism; \
	make clean KOKKOS_OPTIONS='${KOKKOS_OPTIONS}'
