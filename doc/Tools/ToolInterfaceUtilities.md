# ToolSettings / ToolProgrammingInterface

## What are we solving?

In a tools interface, tools need an ability to interact with the programming model, whether that be about telling the programming model which features it supports, or telling the programming model to do something.

In Kokkos, we further desire that tools not depend on some utility library built along with Kokkos. This makes things really difficult, how should the tool tell Kokkos to (for example) fence if it can't link against a specific Kokkos library? We solve this using two utility structures, the `ToolSettings` struct is used to tell Kokkos what features this tool supports, while the `ToolProgrammingInterface` is used to allow the tool to tell Kokkos to do things.

## ToolSettings

The `ToolSettings` struct is certainly the simpler of the two. During initialization, Kokkos invokes the `kokkosp_request_tool_settings` callback, and passes a pointer to a `ToolSettings` struct. This comes fields representing various Kokkos Tools settings. Right now, the only setting is a bool representing whether the tool requires Kokkos to fence before each kernel event. Critically, this field comes prepopulated with default settings. If you ever add a new setting that would allow a tool to change how the interface functions, the default setting should be to make the interface behave in the same way. That way, tools that don't implement `kokkosp_request_tool_settings` see the same behavior they always have. Anyway, after Kokkos invokes that callback, it parses the struct (which the tool may have modified) to determine how it should function. Note that this example talks about bools, but you can put whatever you want in that struct.

## ToolProgrammingInterface

The `ToolProgrammingInterface` is slightly more complicated. Essentially, since the tool can't have a dependency on Kokkos, it needs some other mechanism to communicate with Kokkos. We achieve this by filling the `ToolProgrammingInterface` struct with function pointers the tools can invoke. So, rather than the tool needing to include `Kokkos_Profiling.hpp` somehow, and compile `Kokkos_Profiling.cpp`, and everything they depend on, the tool simply implements `kokkosp_provide_tool_programming_interface`, which accepts a `ToolProgrammingInterface` struct. Then, any time the tool needs to invoke an action, they just use one of the function pointers from that struct.

For an example of usage, see [the fenceless profiling mechanism](FencelessProfilingMechanism.md) section.