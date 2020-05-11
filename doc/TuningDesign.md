# Kokkos Tuning

This is a design document describing the motivation, ideas, design, and prototype implementation of the Kokkos Tuning System

## Motivation

Currently, Kokkos makes a lot of decisions about tuning parameters (CUDA block sizes, different kernel implementations)
by picking an option that results in the best performance for the widest array of applications and architectures at the
time the choice is made. This approach leaves performance on the table, and appears increasingly untenable as the number
of architectures and applications grows, and as software versions change.

The Kokkos team would like to instead open up the ability to set the parameters as part of the tooling system so that
these parameters can be tuned for individual applications across all the architectures they might run on. In order to match the
feel of past Kokkos tooling efforts, we'd like to achieve this with a callback system.

## Ideas

A Kokkos Tuning system should be as small as is wise while achieving the following goals

1. Expose to tools enough data about the _context_ of the running application to tune intelligently. In autotuning terms, decribe the _features_
2. Expose to tools enough data about tuning parameters that they might know how to optimize what they're asked to
3. Expose to applications an interface that they might inform a tool about their current application context
4. Expose to tools the results of their choices
5. No perturbation of Kokkos Core when this system is disabled

Shared among the first three of these goals is a need for some way to describe the semantics of variables (tuning parameters, context variables)
internal to Kokkos or an application to an outside tool.

### Semantics of Variables

I think it's best to talk about the semantics of variables with concrete examples.

Suppose Kokkos wants a tool to choose a block size for it. Suppose all the application context is perfectly understood, that the tool knows
that the application has 10,000,000 particles active and that it's running a kernel called "make_particles_go," which is a parallel_for in
the "cuda" execution space. Even with this knowledge, the tool needs to know several things about what a block size _is_ for this to be generic and practical

1. Is it an integer value? A float? A string? (Type)
2. Relatedly, what are the mathematical semantics which are valid for it? Is it something
for which a list can be sorted? Do the distances between items in a sorted list make sense?
If I divide two values, does the ratio have some meaning? (semantics)
3. What are the valid choices for this value? Is a block size of -128 okay? How about 7? (candidates)

Semantics (as always) are likely the source of the most confusion here, so a bit of detail is good. Here I'm leaning heavily on the field
of statistics to enable tools to do intelligent searching. If ordering doesn't make sense, if a value is "categorical", the only thing
a tool can do is try all possible values for a tuning value. If they're ordered (ordinal), the search can take advantage of this by
using the concept of a directional search. If the distances between elements matter (interval data) you can cheat with things like
bisection. Finally if ratios matter you can play games where you increase by a factor of 10 in your searches. Note that one good point in favor of this design is that it matches up nicely with scikit-opt (a happy accident).

In describing the candidate values in (3), users have two options: sets or ranges. A set has a number of entries of the given type, a range has lower and upper bounds and a step size.

Claim: the combination of context, candidates, semantics, and types gives a tool enough to intelligently explore the search space of
tuning parameters

### Context

Suppose a tool perfectly understands what a block size is. To effectively tune one, it needs to know something about the application.

In a trivial case, the tool knows absolutely nothing other than candidate values for the block size, and tries to make a choice that optimizes across all
invocations of kernels. This isn't _that_ far from what Kokkos does now, so it's not unreasonable for this to produce decent results.
That said, we could quickly add some context from Kokkos, stuff like the name and type of the kernel, the execution space, all with the semantic information described above. That way a tuning tool could differentiate based on all the information available to Kokkos. Going a little further, we could expose this ability to provide context to our applications. What if the tools wasn't just tuning to the fact that the kernel name was "GEMM", but that "matrix_size" was a million? Or that "live_particles" had a
certain value? The more (relevant) context we provide to a tool, the better it will be able to tune.


### Intended Tool Workflow

Okay, so a tool knows what it's tuning, and it knows the context of the application well enough to do clever ML things, all of this with happy semantic information so that everything make . What should a workflow look like? A tool should

1) Listen to declarations about the semantics of context and tuning variables
2) Make tuning decisions
3) Measure their feedback
4) Get better at (2)

The easier we make this loop, the better

## Design

The design of this system is intended to reflect the above ideas with the minimal necessary additions to make the mechanics work. This section is almost entirely describing the small holes in the above descriptions. Variable declaration works exactly as described above, except for two things

1) Each variable is associated with a unique ID at declaration time
2) In addition to allowing "int, float, string" types, we also allow for sets and ranges of the same

(2) is important because it allows us to express interdependency of tuning variables, you can't tune "blockSize.x" and "blockSize.y" independently, the choices intersect [caveat two: interdependent variables must have the same type]. So we wouldn't describe blockSize.x as being between 32 and MAX_BLOCK_SIZE, just like blockSize.y, we would describe "3D_block_size" as being in {1,1,1}, ... {MAX_BLOCK_SIZE,1,1}

Any time a value of a variable is declared (context) or requested (tuning), it is also associated with a context ID that says how long that declaration is valid for. So if a user sees

```c++
declare_value("is_safe_to_push_button",true,contextId(0));
foo();
endContext(contextId(0));
bar();
```

They should know in `bar` that it is no longer safe to push the button. Similarly, if tools have provided tuning values to contextId(0), when contextId(0) ends, that is when the tool takes measurements related to those tuning values and learns things. *For many tools, the first time they see a value associated with a contextId, they'll do a starting measurement, and at endContext they'll stop that measurement*.

The ugliest divergence from design is in the semantics. We would absolutely love to tell users the valid values for a given tuning parameter at variable declaration time. We hate the idea of telling them the valid values on each request for the value of that parameter. Unfortunately the universe is cruel: things can happen outside of Kokkos that make the valid values of a tuning parameter change on each request. Just taking the example of block size

1) Different kernels have different valid values for block size
2) Different invocations of the same kernel can have different values for block size if somebody changes settings
3) We don't know how much worse this gets as we move past block size

So we'll do our best to mitigate the impacts of this, but for now the set of candidate values must be provided every time we request a
value

Otherwise the ideas behind the tuning system translate directly into the design and the implementation

## Implementation

This section describes the implementation.

If you're writing a tool, you care about tool implementation.

If you want tools to know about information from your application, you care about application implementation

If you're a Kokkos developer, you care about the application implementation and Kokkos implementation

### Tool implementation

In the past, tools have responded to the [profiling hooks in Kokkos](https://github.com/kokkos/kokkos-tools/wiki/Profiling-Hooks). This effort adds to that, there are now a few more functions (note that I'm using the C names for types. In general you can replace Kokkos_Tuning_ with Kokkos::Tuning:: in C++ tools)

```c++
void kokkosp_declare_tuning_variable(const char* name, const size_t id, Kokkos_Tuning_VariableInfo info);
```

Declares a tuning variable named `name` with uniqueId `id` and all the semantic information (except candidate values) stored in `info`.

```c++
void kokkosp_declare_context_variable(const char*, const size_t, Kokkos_Tuning_VariableInfo info, Kokkos_Tuning_VariableInfo_SetOrRange);
```

This is much like declaring a tuning variable, except the candidate values of context variables are declared immediately. In cases where they aren't known, `info.valueQuantity` will be set to `kokkos_value_unbounded`. This is fairly common, Kokkos can tell you that `kernel_name` is a string, but we can't tell you what strings a user might provide.

```c++
void kokkosp_request_tuning_variable_values(
    const size_t contextId,
    const size_t numContextVariables, const Kokkos_Tuning_VariableValue* contextVariableValues,
    const size_t numTuningVariables, Kokkos_Tuning_VariableValue* tuningVariableValues, Kokkos_Tuning_VariableInfo_SetOrRange* tuningVariableCandidateValues);
```

Here Kokkos is requesting the values of tuning variables, and most of the meat is here. The contextId tells us the scope across which these variables were used. Often you'll start a measurement/timer and associate it with this ID so that when the contextID ends, you can stop the timer and know how you did.

The next three arguments describe the context you're tuning in. You have the number of context variables, and an array of that size containing their values.

The next four arguments describe the Tuning Variables, first the number, then some default values which you overwrite to change how Kokkos will behave. Finally, you have the candidate values you can choose among.

Critically, as tuningVariableValues comes preloaded with default values, if your function body is `return;` you will not crash Kokkos, only make us use our defaults. If you don't know, you are allowed to punt and let Kokkos do what it would.

```c++
void kokkosp_end_context(const size_t contextId);
```

This simply says that the contextId in the argument is now over. If you provided tuning values associated with that context, those values can now be associated with a result.

First, on the Kokkos side some changes had to happen. If you're writing a tool, skip to Tool Implementation

### App Implementation

For 99% of applications, all you need to do to interact with Kokkos Tuning Tools in your code is nothing. The only exceptions are if you want the tuning to be aware of what's happening in your application (number of particles active, whether different physics are active) if
you think that might change what the Tuning decides. If you're feeling especially brave, you can also use the Tuning interface to tune parameters within your own application. For making people aware of your application context, you need to know about four functions

```c++
size_t getNewVariableId();
```

When you declare a variable, you need to give it a unique ID among variables. You need to use that ID when you're declaring a value for the variable as well. This is just a function to give you an ID.

```c++
size_t getNewContextId();
size_t getCurrentContextId();
```

Similarly, you will associate values with "contexts" in order to decide when a given declaration of a value has gone out of scope. The first gets you a new context ID if you're starting some new set of values. If you need to recover the last context ID so you can append to that context, rather than overwriting it with a new one, you can use `getCurrentContextIDd()`.

```c++
void declareContextVariable(const std::string& variableName, size_t uniqID,
                            VariableInfo info,
                            Kokkos::Tuning::SetOrRange candidate_values);
```

This function tells a tool that you have some variable they should know about when tuning. uniqID field is described above. Info describes the semantics of your variable. This is discussed in great detail under "Semantics of Variables", but you need to say whether the values will be text, int, or float, whether they're categorical, ordinal,interval, or ratio data, and whether the candidate values are "unbounded" (if you don't know the full set of values), a set, or a range. If values are unbounded, you can just pass an empty set [TODO: should we extend the interface with an overload without SetOrRange?]

```c++
void declareContextVariableValues(size_t contextId, size_t count,
                                  VariableValue* values);
```

Here you tell tools the values for your context variables. The contextId is used to later tell when this has gone out of scope, the count is how many variables you're declaring, the uniqIds are an array (of size count) of the unique ID's of those variables, and finally values are the current values of those variables.

```c++
void endContext(size_t contextId);
```

This tells the tool that values from this context are no longer valid. For those who want to declare tuning variables, you only need two more functions.

```c++
void declareTuningVariable(const std::string& variableName, size_t uniqID,
                           VariableInfo info);
```

This is exactly like declareContextVariable, except you don't declare candidate values (you do that when you request values).

```c++
void requestTuningVariableValues(size_t contextId, size_t count,
                                 VariableValue* values,
                                 Kokkos::Tuning::SetOrRange* candidate_values);
```

Here is where you request that the tool give you a value. You need a contextId so that the tool can know when you're done using the value and measure results. The count tells the tool how many variables it's providing values for. Values is an array of your default values for that parameter, it must not crash your program if unchanged. Finally, candidate_values contains the choices the tool might make for that given parameter.

### Kokkos implementation

In the past, Kokkos and Kokkos-tools didn't share source code. Except for a "SpaceHandle" struct which users manually copied to their tools, nothing from Kokkos hit the tools repo, the interface consisted entirely of basic C types. If you read the ideas section, it translates to a lot of structs and enums. Despite my best efforts to minimize them, I think we now need to share some header files with kokkos-tools. Andrew Gaspar did really excellent work making this practical, we have

1) Kokkos_Profiling_C_Interface.h , which is (shockingly) a C interface that everything in Kokkos tools boils down to
2) Kokkos_Profiling_Interface.hpp, nice C++ wrappers around the C so that the C idioms don't hit Kokkos
3) Kokkos_Profiling.[cpp/hpp], which contain things Kokkos needs to implement tooling, but the tools don't need to know about

All of our function pointer initialization and all that mess now go into Kokkos_Profiling.[cpp/hpp], all the types are in the Interface files. The interface files will be shared with kokkos/kokkos-tools.

In terms of build changes, we now install the above .h file, and have a KOKKOS_ENABLE_TUNING option that will be separable from KOKKOS_ENABLE_PROFILING
