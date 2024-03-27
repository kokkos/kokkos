# Fenceless Profiling

It helps to read about [the tool utility structs](ToolInterfaceUtilities.md) to understand some of the mechanisms in use here.

## Why does it matter?

So, in the Kokkos Tools model, we make things really simple for tools. Before Kokkos calls `kokkosp_begin/end_parallel_for` we fence. Because of this, in `kokkosp_begin_parallel_for` the tool just starts a timer, in `kokkosp_end_parallel_for` they stop the timer, and because Kokkos fenced, it's as easy as that to know how long the kernel took. But imagine an asynchronous code, that keeps three streams completely occupied. This fencing essentially serializes the code, slowing it down 3x. In practice, on GPU machines, codes sometimes see 75% slowdowns using tools like the `space-time-stack`. Meanwhile, tools like `APEX` or `Caliper` are smart enough to use CUPTI voodoo to time the kernels without fencing them.

## How is it done?

These tools can use the `ToolSettings` to tell Kokkos "hey, skip the fences before and after launching a `parallel_for`." Then, in `ToolProgrammingInterface` there's a `fence` function. If we're running in a regime where the tool can use some voodoo to time without fencing, they just don't tell Kokkos to fence. If they do require fencing, they tell Kokkos to fence explicitly. 