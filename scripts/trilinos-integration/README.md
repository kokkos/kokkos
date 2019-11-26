![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Kokkos: Promotion Test Debugging

This explains the use (and basic implementation details) of promotion testing
new Kokkos and Kokkos Kernels branches. We first introduce a test for first
validating an existing Trilinos branch that all tests should be passing on (usually the develop, master, or kokkos-promotion branch of Trilinos).
After validating a clean promotion branch, we show how to test the same Trilinos branch
with an updated Kokkoks and Kokkos Kernels.

# Clean Promotion Test
There is a script called `clean_promotion_test` in `scripts/trilinos-integration` that tests a self-contained
Trilinos branch with its default included Kokkos and Kokkos Kernels packages.
The script takes two arguments:
````
./clean_promotion_test <ENV_FILE> <TRILINOS_ROOT>
````
For running CUDA tests on the platform called White, e.g.
````
./clean_promotion_test white_cuda_env.sh $TRILINOS_ROOT
````
Based on the Trilinos path and ENV file, a unique hash `X` is generated.
A CMake configuration is then run in the folder `clean-test-X`.
Output of the configuration will appear in a `config.out`.
If the configuration succeeds, you can `cd` into the folder and run:
````
> source ../<ENV_FILE>
> make -j
````
to validate the build. The configure and build steps are deliberately separated
to allow incremental debugging of each step.

# New Branch Promotion Test
If the clean test is passing, you can now test your updates to Kokkos.
To start the tests, there is a script that now takes four arguments:
````
./config_promotion_test <ENV_FILE> <TRILINOS_ROOT> <KOKKOS_ROOT> <KERNELS_ROOT>
````
where the two additional arguments are the locations of Kokkos and Kokkos Kernels
branches containing updated. Again, a unique hash `X` is generated.
A CMake configuration is then run in the folder `promotion-test-X`.
If the configuration succeeds, you can `cd` into the folder and run:
````
> source ../<ENV_FILE>
> make -j
````
The script uses the the source override feature of the Trilinos build system.
It creates symlinks in the Trilinos folder to your updated Kokkos
and Kokkos Kernels branches. Trilinos is then redirected to build with the updated
Kokkos via a CMake option `-DKokkos_SOURCE_DIR_OVERRIDE=kokkos`.


##### [LICENSE](https://github.com/kokkos/kokkos/blob/master/LICENSE)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.

