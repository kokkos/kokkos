FROM ghcr.io/kokkos/ci-containers/fedora:latest
CMD /bin/bash

WORKDIR /home/kokkos/src/
COPY kokkos/ /home/kokkos/src/kokkos
RUN sudo chown -R kokkos:kokkos kokkos
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=threads
RUN cmake -S kokkos -B kokkos/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_INSTALL_INCLUDEDIR=include/kokkos \
  -DKokkos_ENABLE_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_BENCHMARKS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -DKokkos_ENABLE_HWLOC=ON
RUN cmake --build kokkos/build --parallel 2
RUN ctest --test-dir kokkos/build --output-on-failure
RUN sudo cmake --install kokkos/build
WORKDIR kokkos/example
