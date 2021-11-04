# Kokkos Tools

This is documentation intended to help developers of the Kokkos Tools
subsystem within Kokkos. It was written in November of 2021, concepts
may have drifted between writing and when you're reading this. At the
time I'm leaving, there's moderate awareness of Tools within the team,
but Kevin Huck and Bill Williams are tool developers, on the Kokkos Slack,
and likely to be great resources if this walkthrough doesn't do what you need.

The following topics will be covered:

- [Overview of the design](DesignOverview.md)
- [Code organization](CodeOrganization.md)
- [Tuning](Tuning.md)
  - [Basic concepts](../TuningDesign.md)
  - [MultidimensionalSparseTuningProblem](MultidimensionalSparseTuningProblem.md)
  - [parallel\_for -> begin\_parallel\_for -> tune\_policy -> generic\_tune\_policy](PolicyTuningWorkflow)
- ToolSettings / ToolProgrammingInterface
- Fenceless Profiling Mechanics
- Testing Subsystem
