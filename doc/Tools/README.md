# Kokkos Tools

This is documentation intended to help developers of the Kokkos Tools
subsystem within Kokkos. It was written in November of 2021, concepts
may have drifted between writing and when you're reading this.

The following topics will be covered:

- Overview of the design
- Code organization
- Tuning
  - Basic concepts
  - MultidimensionalSparseTuningProblem
  - parallel\_for -> begin\_parallel\_for -> tune\_policy -> generic\_tune\_policy
- ToolSettings / ToolProgrammingInterface
- Fenceless Profiling Mechanics
- Testing Subsystem
