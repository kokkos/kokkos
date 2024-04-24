//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <cstdio>

// Define a type for a two-dimensional N x N array of double.
// It lives in Kokkos' default memory space.
using matrix_type = Kokkos::View<double**>;

// The "HostMirror" type corresponding to matrix_type above is also a
// two-dimensional N x N array of double. It lives in the host memory space
// corresponding to matrix_type's memory space.
using host_matrix_type = matrix_type::HostMirror;

struct MatrixMultiply {
    matrix_type A, B, C;
    int N;

    MatrixMultiply(matrix_type A_, matrix_type B_, matrix_type C_, int N_)
        : A(A_), B(B_), C(C_), N(N_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j) const {
        double temp = 0.0;
        for (int k = 0; k < N; ++k) {
            temp += A(i, k) * B(k, j);
        }
        C(i, j) = temp;
    }
};

int main() {
    Kokkos::initialize();
    {
        const int N = 256; // Define the size of the matrices.

        // Create matrices A, B, and C in the default execution space.
        matrix_type A("A", N, N), B("B", N, N), C("C", N, N);

        // Create host mirrors of A, B, and C.
        host_matrix_type h_A = Kokkos::create_mirror_view(A);
        host_matrix_type h_B = Kokkos::create_mirror_view(B);
        host_matrix_type h_C = Kokkos::create_mirror_view(C);

        // Initialize matrices A and B on the host.
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                h_A(i, j) = 1.0 * (i + 1);
                h_B(i, j) = 1.0 * (j + 1);
            }
        }

        // Copy initialized matrices from the host to the device.
        Kokkos::deep_copy(A, h_A);
        Kokkos::deep_copy(B, h_B);

        // Perform matrix multiplication using parallel MDRangePolicy.
        using MDRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for("MatrixMultiply", MDRangePolicy({0, 0}, {N, N}),
                             MatrixMultiply(A, B, C, N));

        Kokkos::deep_copy(h_C, C);  // Copy from host to device.

        // Use printf to output some elements of matrix C to verify correctness.
        printf("C[0][0] = %f\n", h_C(0, 0));
        printf("C[N-1][N-1] = %f\n", h_C(N-1, N-1));
    }
    Kokkos::finalize();
    return 0;
}
