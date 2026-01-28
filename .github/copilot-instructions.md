# Kokkos Copilot Instructions

## What is Kokkos?
C++ performance portability library for HPC with parallel execution abstractions. Supports CUDA, HIP, SYCL, OpenMP, Threads, Serial backends. ~12MB codebase, C++20 minimum, CMake 3.22+. **Primary branch: `develop`** (not main).

## Build Instructions (CRITICAL)

### Standard Build - Always Use Separate Directory
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON ..
cmake --build . --parallel $(nproc)  # ~20s minimal, 3-10min with tests
ctest --output-on-failure
```

**In-source builds FORBIDDEN** - CMake errors immediately.

### Key Options
**Backends** (≥1 required, SERIAL auto-enabled if none): `-DKokkos_ENABLE_<SERIAL|OPENMP|THREADS|CUDA|HIP|SYCL>=ON`
**Dev flags**: `-DKokkos_ENABLE_TESTS=ON`, `-DKokkos_ENABLE_COMPILER_WARNINGS=ON`, `-DCMAKE_CXX_FLAGS="-Werror"` (CI uses this)
**Standards**: `-DCMAKE_CXX_STANDARD=20` (or 23)

### CI-Compatible Build
```bash
cmake -B builddir -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-Werror" -DCMAKE_CXX_STANDARD=20 \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=ON \
      -DKokkos_ENABLE_COMPILER_WARNINGS=ON
cmake --build builddir --parallel $(nproc)
ctest --test-dir builddir --timeout 2000 --output-on-failure
```

## Code Quality (Pre-commit Required)

**ALWAYS run before commit:** `pre-commit run --all-files` (installs: `pip install pre-commit && pre-commit install`)

**Checks enforced:**
- **clang-format v16** (Google style, `SortIncludes: false` - never reorder includes!)
- **cmake-format** (auto-formats CMake files)
- **License headers** (auto-inserted from `scripts/license_header.txt`, C++: `//`, C: `/* */`)
- YAML, trailing whitespace, EOF fixes

**clang-tidy** (static analysis, used in CI):
```bash
cmake -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-warnings-as-errors=*" ..
```
Checks: bugprone-*, modernize-use-nullptr/using, kokkos-* (config: `.clang-tidy`)

### Naming Style Guidelines for Kokkos Development

To maintain a consistent codebase, the following naming conventions should be followed:

1. **Classes**:
   - Use `CamelCase` style.
   - Keep names descriptive but concise (e.g., `ExecutionSpace`, `MemoryManager`).

2. **Class Members**:
   - Use `snake_case` style.
   - Prefix private data members with `m_` (e.g., `m_value` for internal data).

3. **Functions**:
   - Use `snake_case` style for both member and free functions.
   - Names should clearly describe the function's purpose (e.g., `get_functor`, `register_event`).

4. **Template Parameters**:
   - Use descriptive names in `CamelCase` (e.g., `Data`, `Handler`).

5. **Macros**:
   - Use `ALL_UPPERCASE` with underscores (e.g., `KOKKOS_VERSION`, `KOKKOS_ENABLE_DEBUG`).

## Repository Structure

```
kokkos/
├── core/          # Core library (7.7M, ~295 headers, ~38 cpp) - PRIMARY
│   ├── src/       # Implementation
│   └── unit_test/ # Extensive tests
├── algorithms/    # Kokkos Algorithms (2.3M)
├── containers/    # Kokkos Containers (904K)
├── simd/          # SIMD abstractions (1.1M)
├── cmake/         # Build system (kokkos_enable_devices.cmake, kokkos_functions.cmake)
├── bin/           # nvcc_wrapper (CUDA builds), kokkos_launch_compiler
├── example/       # Integration tests (build_cmake_installed/, tutorial/)
├── tpls/          # Third-party: desul (atomics), gtest, mdspan (DO NOT MODIFY)
├── .github/workflows/  # CI: continuous-integration-{linux,windows,osx}.yml, pre-commit.yml
├── CMakeLists.txt # Root config (version, backends, options)
├── .clang-format  # v16, Google style, SortIncludes: false
└── .pre-commit-config.yaml
```

**Key files:** `CMakeLists.txt`, `cmake/kokkos_enable_devices.cmake` (backends), `.clang-{
  format, tidy}`, `docs/CONTRIBUTING.md`, `bin/nvcc_wrapper`

## CI/CD Workflows

**Main workflows** (.github/workflows/):
1. **continuous-integration-linux.yml** - Primary (g++, clang++, icpx matrix, SERIAL/OPENMP/THREADS, clang-tidy, ASan/UBSan)
2. **continuous-integration-smoketest.yml** - Fast check (clang++, `-Werror`, runs every PR, MUST PASS)
3. **pre-commit.yml** - Formatting/linting (MUST PASS or format locally first)

**Replicate CI locally:**
```bash
cmake -B builddir -DCMAKE_CXX_FLAGS="-Werror" \
      -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-warnings-as-errors=*" \
      -DKokkos_ENABLE_COMPILER_WARNINGS=ON -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_SERIAL=ON
cmake --build builddir --parallel $(nproc) && ctest --test-dir builddir --timeout 2000 --output-on-failure
pre-commit run --all-files
```

## Testing

```bash
ctest --output-on-failure              # All tests, show failures
ctest -R CoreUnitTest --timeout 2000   # Pattern match with timeout (CI uses 2000s)
ctest -j$(nproc)                       # Parallel execution
```

**Test structure:** `core/unit_test/` (main), `algorithms/unit_tests/`, `containers/unit_tests/`, `simd/unit_tests/`, `example/` (integration)
**Naming:** `Kokkos_<Component>UnitTest_<Backend>_<TestName>`
**Note:** Tests are backend-specific. Some disabled for certain backends (see `# FIXME_<BACKEND>` comments in CMakeLists.txt).

## Critical Rules

### ❌ NEVER:
1. In-source builds (CMake blocks this)
2. Commit without license headers (pre-commit fails)
3. Commit unformatted code (pre-commit fails)
4. Modify `tpls/` files (third-party, formatting-excluded)
5. Reorder includes (SortIncludes disabled intentionally)
6. Add submodules (pre-commit forbids)
7. Target `main` branch (use `develop`)

### ✅ ALWAYS:
1. Build: `mkdir build && cd build` (separate directory)
2. Pre-commit: `pre-commit run --all-files` before commit
3. Test with `-Werror`: `-DCMAKE_CXX_FLAGS="-Werror"`
4. Enable backend (SERIAL auto-enabled if none)
5. Use C++20+
6. Run tests: `ctest --output-on-failure`
7. PR to `develop`, not `main`

### Common Issues:
- **clang-format version mismatch**: CI uses v16.0.0
- **Build timeout**: Use parallel build `-j$(nproc)`, allow 3-10min for tests
- **CUDA builds**: Requires `bin/nvcc_wrapper`
- **Default backend**: SERIAL enabled automatically if none specified

---

**Trust these instructions.** Search codebase only if info incomplete/incorrect. Advanced build details: `cmake/README.md`.
