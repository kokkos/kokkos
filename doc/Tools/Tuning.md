# Tuning

This section describes the autotuning capabilities in Kokkos, in terms of the facilities available, what goes where, and the like. As always, the code organization is a bit rough, do not be shy about moving certain classes of facilities into specific files.

## Callbacks

The concepts of all of these callbacks are described in the TuningDesign[../TuningDesign] document, this section will describe layout.

As in Profiling, types are defined in Profiling_C_Interface.h, given namespace in Profiling.hpp, and callbacks are implemented in Profiling.cpp. Then, much of the higher-level reasoning ("should I tune a RangePolicy?") all happens in Kokkos_Tools_Generic.hpp. Critically, a lot of magic happens because `begin_parallel_[for/scan/reduce]` start a tuning context, and `end_parallel_[for/scan/reduce]` end a tuning context, and declare the names of the kernel as a feature. This is why team size tuners, mdrange tuners, and pretty much every kernel tuner work without the user making any changes.

In `Kokkos_Tuners.hpp`, there are a few shortcuts available to make your life easier. MultidimensionalSparseTuningProblem is somewhat baroque software architecture, so [it gets its own section](MultidimensionalSparseTuningProblem). But the big one is that there's a `CategoricalTuner`. If you ever just want to pick amongst categorical options, I recommend using that, it's a nice convenience layer.

Additionally, one confusing thing is how we progress from a user calling parallel_for,
to the TeamPolicy in that parallel_for suddenly having its team_size changed. I describe that in the [policy tuning workflow](PolicyTuningWorkflow) section.