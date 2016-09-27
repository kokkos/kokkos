build:
	mkdir -p 01_thread_teams
	cd ./01_thread_teams; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 01_thread_teams_lambda
	cd ./01_thread_teams_lambda; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams_lambda/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 02_nested_parallel_for
	cd ./02_nested_parallel_for; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/02_nested_parallel_for/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 03_vectorization
	cd ./03_vectorization; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/03_vectorization/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 04_team_scan
	cd ./04_team_scan; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/04_team_scan/Makefile ${KOKKOS_OPTIONS}

build-insource:
	cd ./01_thread_teams; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./01_thread_teams_lambda; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./02_nested_parallel_for; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./03_vectorization; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./04_team_scan; \
	make build -j 4 ${KOKKOS_OPTIONS}
test:
	cd ./01_thread_teams; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams/Makefile ${KOKKOS_OPTIONS}
	cd ./01_thread_teams_lambda; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./02_nested_parallel_for; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/02_nested_parallel_for/Makefile ${KOKKOS_OPTIONS}
	cd ./03_vectorization; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/03_vectorization/Makefile ${KOKKOS_OPTIONS}
	cd ./04_team_scan; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/04_team_scan/Makefile ${KOKKOS_OPTIONS}

test-insource:
	cd ./01_thread_teams; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./01_thread_teams_lambda; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./02_nested_parallel_for; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./03_vectorization; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./04_team_scan; \
	make test -j 4 ${KOKKOS_OPTIONS}
clean:
	cd ./01_thread_teams; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams/Makefile ${KOKKOS_OPTIONS}
	cd ./01_thread_teams_lambda; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/01_thread_teams_lambda/Makefile ${KOKKOS_OPTIONS}
	cd ./02_nested_parallel_for; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/02_nested_parallel_for/Makefile ${KOKKOS_OPTIONS}
	cd ./03_vectorization; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/03_vectorization/Makefile ${KOKKOS_OPTIONS}
	cd ./04_team_scan; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Hierarchical_Parallelism/04_team_scan/Makefile ${KOKKOS_OPTIONS}

clean-insource:
	cd ./01_thread_teams; \
	make clean ${KOKKOS_OPTIONS}
	cd ./01_thread_teams_lambda; \
	make clean ${KOKKOS_OPTIONS}
	cd ./02_nested_parallel_for; \
	make clean ${KOKKOS_OPTIONS}
	cd ./03_vectorization; \
	make clean ${KOKKOS_OPTIONS}
	cd ./04_team_scan; \
	make clean ${KOKKOS_OPTIONS}
