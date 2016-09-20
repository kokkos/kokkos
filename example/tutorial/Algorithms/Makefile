build:
	mkdir -p 01_random_numbers
	cd ./01_random_numbers; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Algorithms/01_random_numbers/Makefile ${KOKKOS_OPTIONS}

build-insource:
	cd ./01_random_numbers; \
	make build -j 4 ${KOKKOS_OPTIONS}
test:
	cd ./01_random_numbers; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Algorithms/01_random_numbers/Makefile ${KOKKOS_OPTIONS}

test-insource:
	cd ./01_random_numbers; \
	make test -j 4 ${KOKKOS_OPTIONS}
clean:
	cd ./01_random_numbers; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Algorithms/01_random_numbers/Makefile ${KOKKOS_OPTIONS}

clean-insource:
	cd ./01_random_numbers; \
	make clean ${KOKKOS_OPTIONS}
