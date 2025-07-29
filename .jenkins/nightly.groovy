pipeline {
    agent none

    environment {
        SPACK_CDASH_ARGS="--cdash-upload-url=https://my.cdash.org/submit.php?project=Kokkos --cdash-track=Nightly --cdash-site=ornl-jenkins"
    }

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

                          export CDASH_ARGS="${SPACK_CDASH_ARGS} --cdash-build=spack-serial"
                          rm -rf spack && \
                          git clone https://github.com/spack/spack.git && \
                          . ./spack/share/spack/setup-env.sh && \
                          spack install --only=dependencies kokkos@develop+tests && \
                          spack install --only=package ${CDASH_ARGS} kokkos@develop+tests && \
                          spack load cmake && \
                          spack test run ${CDASH_ARGS} kokkos && \
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

                          export CDASH_ARGS="${SPACK_CDASH_ARGS} --cdash-build=spack-cuda"
                          rm -rf spack && \
                          git clone https://github.com/spack/spack.git && \
                          . ./spack/share/spack/setup-env.sh && \
                          spack install --only=dependencies kokkos@develop+cuda+wrapper+tests cuda_arch=80 ^cuda@12.1.0 && \
                          spack install --only=package ${CDASH_ARGS} kokkos@develop+cuda+wrapper+tests cuda_arch=80 ^cuda@12.1.0 && \
                          spack load cmake  && \
                          spack load kokkos-nvcc-wrapper && \
                          spack load cuda && \
                          spack load kokkos && \
                          spack test run ${CDASH_ARGS} kokkos && \
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

                          export CMAKE_BUILD_PARALLEL_LEVEL=8 && \
                          export ENV_CMAKE_OPTIONS="" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_BUILD_TYPE=Release" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_STANDARD=26" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_FLAGS=-Werror" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ARCH_NATIVE=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_COMPILER_WARNINGS=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_BENCHMARKS=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_EXAMPLES=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_TESTS=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_DEPRECATED_CODE_4=ON" && \
                          export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_SERIAL=ON" && \
                          ctest -VV -D CDASH_MODEL="Nightly" -D CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS}" -S scripts/CTestRun.cmake -D CTEST_SITE="ornl-jenkins" -D CTEST_BUILD_NAME="GCC-15-CXX26"
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
                        sh '''export CMAKE_BUILD_PARALLEL_LEVEL=16 && \
                              export ENV_CMAKE_OPTIONS="" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_BUILD_TYPE=RelWithDebInfo" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_COMPILER=hipcc" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_STANDARD=20" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_FLAGS='-Werror -Wno-unused-command-line-argument'" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ARCH_NATIVE=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_COMPILER_WARNINGS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_DEPRECATED_CODE_4=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_TESTS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_BENCHMARKS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_EXAMPLES=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_HIP=ON" && \
                              ctest -VV -D CDASH_MODEL="Nightly" -D CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS}" -S scripts/CTestRun.cmake -D CTEST_SITE="ornl-jenkins" -D CTEST_BUILD_NAME="HIP-ROCM-6.4-MI100-RDC-CXX20"
                              '''
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
                        sh '''export CMAKE_BUILD_PARALLEL_LEVEL=16 && \
                              export ENV_CMAKE_OPTIONS="" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_BUILD_TYPE=RelWithDebInfo" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_COMPILER=hipcc" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_STANDARD=23" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DCMAKE_CXX_FLAGS='-Werror -Wno-unused-command-line-argument'" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ARCH_NATIVE=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_COMPILER_WARNINGS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_DEPRECATED_CODE_4=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_TESTS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_BENCHMARKS=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_EXAMPLES=ON" && \
                              export ENV_CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS};-DKokkos_ENABLE_HIP=ON" && \
                              ctest -VV -D CDASH_MODEL="Nightly" -D CMAKE_OPTIONS="${ENV_CMAKE_OPTIONS}" -S scripts/CTestRun.cmake -D CTEST_SITE="ornl-jenkins" -D CTEST_BUILD_NAME="HIP-ROCM-6.4-MI210-CXX23"
                              '''
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
