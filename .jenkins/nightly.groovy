pipeline {
    agent none

    options {
        timeout(time: 6, unit: 'HOURS')
    }

    stages {
        stage('Build') {
            parallel {
                stage('spack-serial') {
                    agent {
                        docker {
                          image 'ubuntu:22.04'
                          label 'docker'
                        }
                    }
                    steps {
                        sh '''
                          DEBIAN_FRONTEND=noninteractive && \
                          apt-get update && apt-get upgrade -y && apt-get install -y \
                          build-essential \
                          wget \
                          git \
                          bc \
                          python3-dev \
                          && \
                          apt-get clean && rm -rf /var/lib/apt/lists/*

                          rm -rf spack && \
                          git clone https://github.com/spack/spack.git && \
                          . ./spack/share/spack/setup-env.sh && \
                          spack install kokkos@develop+tests && \
                          spack load cmake && \
                          spack test run kokkos && \
                          spack test results -l
                          '''
                    }      
                }   
                stage('spack-cuda') {
                    agent {
                        docker {
                          image 'nvidia/cuda:12.1.0-devel-ubuntu22.04'
                          label 'nvidia-docker && ampere'
                        }
                    }
                    steps {
                        sh '''
                          DEBIAN_FRONTEND=noninteractive && \
                          apt-get update && apt-get upgrade -y && apt-get install -y \
                          build-essential \
                          wget \
                          git \
                          bc \
                          python3-dev \
                          gfortran \
                          && \
                          apt-get clean && rm -rf /var/lib/apt/lists/*

                          rm -rf spack && \
                          git clone https://github.com/spack/spack.git && \
                          . ./spack/share/spack/setup-env.sh && \
                          spack install kokkos@develop+cuda+wrapper+tests cuda_arch=80 ^cuda@12.1.0 && \
                          spack load cmake  && \
                          spack load kokkos-nvcc-wrapper && \
                          spack load cuda && \
                          spack load kokkos && \
                          spack test run kokkos && \
                          spack test results -l
                          '''
                    }      
                }   
                stage('GCC-15-CXX26') {
                    agent {
                        docker {
                            image 'gcc:15.1'
                            label 'docker'
                        }
                    }
                    steps {
                        sh '''
                          wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.sh && \
                          chmod +x cmake-3.30.0-linux-x86_64.sh && ./cmake-3.30.0-linux-x86_64.sh --skip-license --prefix=/usr

                          rm -rf build && mkdir -p build && cd build && \
                          cmake \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DCMAKE_CXX_STANDARD=26 \
                            -DCMAKE_CXX_FLAGS=-Werror \
                            -DKokkos_ARCH_NATIVE=ON \
                            -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                            -DKokkos_ENABLE_BENCHMARKS=ON \
                            -DKokkos_ENABLE_EXAMPLES=ON \
                            -DKokkos_ENABLE_TESTS=ON \
                            -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                            -DKokkos_ENABLE_SERIAL=ON \
                          .. && \
                          make -j8 && ctest --no-compress-output -T Test --verbose
                          '''
                    }
                    post {
                        always {
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('HIP-ROCM-6.4-MI100-RDC-CXX20') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-24.04:6.4.1-complete'
                            label 'rocm-docker && AMD_Radeon_Instinct_MI100'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        // FIXME Test returns a wrong value
                        GTEST_FILTER = '-hip_hostpinned.view_allocation_large_rank'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=hipcc \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unused-command-line-argument" \
                                -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_EXAMPLES=ON \
                                -DKokkos_ENABLE_HIP=ON \
                              .. && \
                              make -j16 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('HIP-ROCM-6.4-MI210-CXX23') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-24.04:6.4.1-complete --build-arg CMAKE_VERSION=3.31.3'
                            label 'rocm-docker && AMD_Radeon_Instinct_MI210'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        // FIXME Test returns a wrong value
                        GTEST_FILTER = '-hip_hostpinned.view_allocation_large_rank'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=hipcc \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unused-command-line-argument" \
                                -DCMAKE_CXX_STANDARD=23 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_HIP=ON \
                              .. && \
                              make -j16 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
            }
        }
    }
}
