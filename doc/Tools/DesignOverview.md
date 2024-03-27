# Design Overview

This document covers the high-level design goals and implementation of the Kokkos Tools subsystem.

## History

Much of the design of Kokkos Tools has influences from its history. As Kokkos was being developed, the developers wanted ways to check the performance of Kokkos code in a platform and backend-agnostic way. Suppose someone had some clever new optimization in Kokkos itself, and wanted to check whether this was (a) brilliant or (b) deeply stupid. As this capability was developed, it became more and more clear that these facilities were useful to developers of Kokkos applications. 

## Goals

One of the early goals of the Kokkos Tools subsystem was to make it really easy to develop simple tools for simple tasks. Specifically, it was seen as deeply obnoxious to require

- A tool to need some version-dependent Kokkos_Make_The_Tool_Work.hpp file to achieve simple goals
- An app developer to recompile their code to change or add a tool
- Any noticeable overhead when a tool was not in use

The first is for the convenience of tool developers, and it is critically important. As of today, we _do_ have header files for some advanced Kokkos Tools functionality in the Kokkos_Profiling_C_Interface.h and Kokkos_Profiling.hpp files, but these are backwards compatible for almost five years. If suddenly TAU needs to know whether it's being used by Kokkos 3.6 or Kokkos 3.6.1, we will have failed in a promise to tool developers. This matters because right now, Tools have to guard profiling layers like OMPT behind an "ENABLE_OMPT" build config option, which defaults to off. Meanwhile, an increasing set of tools either have no build option to disable Kokkos Tools (see: TAU), or at least to default to building Kokkos Tools (see: Caliper). We are also discussing with vendors (see Christian Trott for details) the option of them supporting Kokkos Tools on a first-party basis, something that _won't_ happen if we aren't backwards-compatible.

The second comes from convenience to Kokkos app developers, specifically those who write million line codes that take days to compile. It will be extremely tempting to you to say "ahhhh, if I have such-and-such advanced C++ template metaprogramming in the tool and require a recompile, I can change the world," but if it takes developers six hours to get feedback from a tool, it won't matter. This isn't a caution against doing _anything_ that requires a recompile, the resilience work is an effective example of tooling that requires recompilation, but the tools layer should not require recompilation to change or add tools, if such a design is feasible.

The final goal is simplest, but extremely important. If an app developer sees a slowdown from Kokkos Tools without actually having a tool loaded, they're going to ask us to remove the Kokkos Tools layer, which would be very sad. Making sure that we keep overhead near-zero when not in use is critical. As an example of how this can be complex, consider the name passed to begin\_parallel\_x events. If a user doesn't provide a name, Kokkos generates one. Critically, Kokkos only generates that name _if a tool is loaded_, even the overhead of generating the name would be too much. Further, our current implementation is inefficient, it should only generate the name if the loaded tool implements begin\_parallel\_x. Keeping an eye on overheads is truly critical, our users are awesome and will be friendly about it if they see a slowdown so long as we fix it, but there is likely a breaking point where they ask to compile out the tools layer, which is an absolute nightmare.

## Implementation

The fundamental design in Kokkos Tools is a function-pointer based callback interface. This arises largely out of the low-overhead goal. Consider the following implementation sketch:

```c++
begin_function_ptr begin;
end_function_ptr end;
void parallel_for(const std::string& label, boring_parallelism_stuff stuff) {
  if(begin != nullptr){
      fence();
      (*begin)(label);
  }
  // do boring parallelism things
  if(end != nullptr){
      fence();
      (*end)();
  }
}
```

What's neat about this design is that if the begin and end function pointers are null, this amounts to

```console
(compare ptr to zero, super fast)
run the code
(compare ptr to zero, super fast)
```

And a branch predictor on CPUs from the 1900s forward should be able to pick up the pattern of that "compare to zero" always being true. This is the rough mechanism we use to ensure low overheads when tools aren't in use.

But what about not requiring recompilation to swap out tools? For this we primarily use the dlopen/dlsym tricks on a standard Linux system. Unfortunately, I don't know good docs on dlsym and dlopen, here's the [man page](https://pubs.opengroup.org/onlinepubs/009695399/functions/dlsym.html), but just look for people at a national lab or university with a haunted look in their eyes and you've probably found the people who have to wrestle with shared linkers, they'll know dlsym. The TL;DR is that dlopen opens a shared library and gives you a handle you can pass to dlsym along with a function name to retrieve a function pointer _from_ that shared library. Expanding on our earlier example, this looks roughly like:

```c++
begin_function_ptr begin;
end_function_ptr end;
void initialize(){
    std::string profile_library = getenv("KOKKOS_PROFILE_LIBRARY");
    void* lookup_handle = dlopen(profile_library);
    begin = dlsym(lookup_handle, "kokkosp_begin_parallel_for");
    end   = dlsym(lookup_handle, "kokkosp_end_parallel_for");
}
void parallel_for(const std::string& label, boring_parallelism_stuff stuff) {
  if(begin != nullptr){
      fence();
      (*begin)(label);
  }
  // do boring parallelism things
  if(end != nullptr){
      fence();
      (*end)();
  }
}
```

Now, if a user sets `KOKKOS_PROFILE_LIBRARY` before running their application, we will open up that library, fill out the begin and end function pointers, and do clever tooling things. If they don't set the environment variable, those function pointers will remain `NULL` and we will fall back on the super fast comparisons to nullptr. Critically, this `dlsym` mechanism strongly encourages that all callback signatures you look up with dlsym should be compatible with the `C` language, having a callback with `C++` types in their signature is dangerous in 2021 due to concerns about C++ ABI (look for people at labs and unis who have given up on life for an in-depth treatment of C++ ABI). If it's 2030 and Clang is the only C++ compiler, and machines have exactly one libstdc++ installed, and we all ride unicorns to work, it _might_ be safe to revisit C++ in callback signatures.

Note: in addition to the `dlsym` mechanism, we also allow users to call functions to set these function pointers, for those barbarous places (Windows modulo WSL) where dlsym is not supported.

The first goal, not requiring versioned tool headers, is actually simpler. Originally, we just made sure that the function signatures we support used simple types, `uint64_t`, stuff you don't need Kokkos headers for. Since then we've loosened the requirement, and developed headers for complex tasks in the Kokkos Tools system (think autotuning). For an exploration of that work, see the [code organization guide](CodeOrganization.md)

### Reference: list of events

For an exhaustive and _definitely_ up-to-date listing of supported events, see the [Profiling Hooks](https://github.com/kokkos/kokkos-tools/wiki/Profiling-Hooks) wiki page

## A Tool's Perspective: What do Tools Get From Kokkos, What Comes from Other Sources?

TODO: Request from Bill Williams, currently a stub section