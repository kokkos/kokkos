# SPDX-FileCopyrightText: Copyright 2026 Antmicro
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler options for the supported host backends."""

CXX_COPTS = select({
    "//bazel:clang": ["-std=c++20"],
    "//bazel:gcc": ["-std=c++20"],
}, no_match_error = "Kokkos Bazel builds currently support GCC and Clang toolchains.") + select({
    "//:openmp": [],
    "//bazel:openmp_apple_clang": ["-Xpreprocessor", "-fopenmp"],
    "//bazel:openmp_clang": ["-fopenmp"],
    "//bazel:openmp_gcc": ["-fopenmp"],
    "//conditions:default": [],
})

LINKOPTS = select({
    "//bazel:linux": ["-ldl", "-pthread"],
    "//bazel:macos": ["-pthread"],
    "//conditions:default": [],
}) + select({
    "//:openmp": [],
    "//bazel:openmp_apple_clang": ["-lomp"],
    "//bazel:openmp_clang": [],
    "//bazel:openmp_gcc": ["-fopenmp"],
    "//conditions:default": [],
})

SUPPORTED_PLATFORMS = select({
    "//bazel:linux": [],
    "//bazel:macos": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

OPENMP_DEPS = select({
    "//bazel:openmp_clang": ["@openmp"],
    "//conditions:default": [],
})
