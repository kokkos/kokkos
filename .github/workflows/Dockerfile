FROM ghcr.io/kokkos/ci-containers/fedora:latest

WORKDIR /home/kokkos/src/
COPY kokkos/ /home/kokkos/src/kokkos
RUN sudo chown -R kokkos:kokkos kokkos
RUN cmake -S kokkos -B kokkos/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DKokkos_ENABLE_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_BENCHMARKS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
  -DKokkos_ENABLE_HWLOC=ON
RUN cmake --build kokkos/build --parallel 2
RUN ctest --test-dir kokkos/build --output-on-failure
RUN sudo cmake --install kokkos/build
