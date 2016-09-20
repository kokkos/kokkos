build:
	mkdir -p 01_data_layouts
	cd ./01_data_layouts; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/01_data_layouts/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 02_memory_traits
	cd ./02_memory_traits; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/02_memory_traits/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 03_subviews
	cd ./03_subviews; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/03_subviews/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 04_dualviews
	cd ./04_dualviews; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/04_dualviews/Makefile ${KOKKOS_OPTIONS}
	mkdir -p 05_NVIDIA_UVM
	cd ./05_NVIDIA_UVM; \
	make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/05_NVIDIA_UVM/Makefile ${KOKKOS_OPTIONS}
	#mkdir -p 06_AtomicViews
	#cd ./06_AtomicViews; \
	#make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/06_AtomicViews/Makefile ${KOKKOS_OPTIONS}
	#mkdir -p 07_Overlapping_DeepCopy
	#cd ./07_Overlapping_DeepCopy; \
	#make build -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/07_Overlapping_DeepCopy/Makefile ${KOKKOS_OPTIONS}

build-insource:
	cd ./01_data_layouts; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./02_memory_traits; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./03_subviews; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./04_dualviews; \
	make build -j 4 ${KOKKOS_OPTIONS}
	cd ./05_NVIDIA_UVM; \
	make build -j 4 ${KOKKOS_OPTIONS}
	#cd ./06_AtomicViews; \
	#make build -j 4 ${KOKKOS_OPTIONS}
	#cd ./07_Overlapping_DeepCopy; \
	#make build -j 4 ${KOKKOS_OPTIONS}
test:
	cd ./01_data_layouts; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/01_data_layouts/Makefile ${KOKKOS_OPTIONS}
	cd ./02_memory_traits; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/02_memory_traits/Makefile ${KOKKOS_OPTIONS}
	cd ./03_subviews; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/03_subviews/Makefile ${KOKKOS_OPTIONS}
	cd ./04_dualviews; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/04_dualviews/Makefile ${KOKKOS_OPTIONS}
	cd ./05_NVIDIA_UVM; \
	make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/05_NVIDIA_UVM/Makefile ${KOKKOS_OPTIONS}
	#cd ./06_AtomicViews; \
	#make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/06_AtomicViews/Makefile ${KOKKOS_OPTIONS}
	#cd ./07_Overlapping_DeepCopy; \
	#make test -j 4 -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/07_Overlapping_DeepCopy/Makefile ${KOKKOS_OPTIONS}

test-insource:
	cd ./01_data_layouts; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./02_memory_traits; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./03_subviews; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./04_dualviews; \
	make test -j 4 ${KOKKOS_OPTIONS}
	cd ./05_NVIDIA_UVM; \
	make test -j 4 ${KOKKOS_OPTIONS}
	#cd ./06_AtomicViews; \
	#make test -j 4 ${KOKKOS_OPTIONS}
	#cd ./07_Overlapping_DeepCopy; \
	#make test -j 4 ${KOKKOS_OPTIONS}
clean:
	cd ./01_data_layouts; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/01_data_layouts/Makefile ${KOKKOS_OPTIONS}
	cd ./02_memory_traits; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/02_memory_traits/Makefile ${KOKKOS_OPTIONS}
	cd ./03_subviews; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/03_subviews/Makefile ${KOKKOS_OPTIONS}
	cd ./04_dualviews; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/04_dualviews/Makefile ${KOKKOS_OPTIONS}
	cd ./05_NVIDIA_UVM; \
	make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/05_NVIDIA_UVM/Makefile ${KOKKOS_OPTIONS}
	#cd ./06_AtomicViews; \
	#make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/06_AtomicViews/Makefile ${KOKKOS_OPTIONS}
	#cd ./07_Overlapping_DeepCopy; \
	#make clean -f ${KOKKOS_PATH}/example/tutorial/Advanced_Views/07_Overlapping_DeepCopy/Makefile ${KOKKOS_OPTIONS}

clean-insource:
	cd ./01_data_layouts; \
	make clean ${KOKKOS_OPTIONS}
	cd ./02_memory_traits; \
	make clean ${KOKKOS_OPTIONS}
	cd ./03_subviews; \
	make clean ${KOKKOS_OPTIONS}
	cd ./04_dualviews; \
	make clean ${KOKKOS_OPTIONS}
	cd ./05_NVIDIA_UVM; \
	make clean ${KOKKOS_OPTIONS}
	#cd ./06_AtomicViews; \
	#make clean ${KOKKOS_OPTIONS}
	#cd ./07_Overlapping_DeepCopy; \
	#make clean ${KOKKOS_OPTIONS}
