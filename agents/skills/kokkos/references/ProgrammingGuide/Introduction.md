# Introduction

The field of High Performance Computing (HPC) is on the verge of entering a new
era. The need for a fundamental change comes from many angles including the growing
acceptance that rates of pure computation (often called FLOP/s) are a poor single
optimization goal for scientific workloads, as well as practical challenges in the form
of producing energy efficient and cost-efficient processors. Since the convergence on
the Message-Passing Interface (MPI) standard in the mid 1990s, application developers
have enjoyed a seemingly static view of the underlying machine - that of a distributed
collection of homogeneous nodes executing in collaboration. However, after almost two
decades of dominance, the sole use of MPI to derive parallelism is acting as a potential
limiter to improved future performance. While we expect MPI to continue to function
as the basic mechanism for communication between compute nodes for the immediate
future, additional parallelism is likely to be required on the computing node itself if high
performance and efficiency goals are to be realized.

In reviewing potential options for the computing nodes of the future the reader
might fall upon three broad categories of computing devices:

* the multicore processor with powerful serial performance, optimized to reduce operation latency, 

* many-core processors with low to medium powered compute cores that are designed to extend the multicore concept toward throughput based computation,

* and finally, the general-purpose graphics processor unit (GP-GPU, or often, GPU) which is a much more numerous collection of low powered cores designed to tolerate long latencies but provide performance through a much higher degree of parallelism and computational throughput. 

Any combination of these options might also be combined in the future.

The diversity of the options available for processor selection raises an interesting question as to how these should be programmed. In part due to their heritage, but also
due to their optimized designs, each of these hardware types offers a different programming solution and a different set of guidelines by which to write applications for highest
performance. Options available today include a number of shared memory approaches
such as OpenMP, Cilk+, Thread Building Blocks (TBB) as well as Linux p-threads. To
target both contemporary multi/many-core processors and GPUs technologies options
such as OpenMP, OpenACC and OpenCL might be used. Finally, for highest performance 
on GPUs a programming model such as CUDA may be selected. Such a variety
of options poses a problem to the application developer of today which is reminiscent
of the challenges before MPI became the default communication library:

* which model should be selected to provide portability across hardware solutions?

* and also provide high performance across each class of processor?

* and protect algorithm investment into the future?

None of the models listed above have been able to provide practical solutions
to these questions.

The Kokkos programming model described in this programming guide seeks to address these concerns by providing an abstraction of both computation and application data allocation and layout. These abstraction layers are designed specifically to isolate software developers from fluctuation and diversity in hardware details yet provide portability and high levels of performance across many architectures.

This guide provides an introduction to the motivations of developing such an abstraction library, a coverage of the programming model and its implementation as an embedded C++ library requiring no additional modifications to the base C++ language. As such it should be seen as an introduction for new users and as a reference for application developers who are already employing Kokkos in their applications. Finally, supplementary tutorial examples are included as part of the Kokkos software release to help users experiment and explore the programming model through a gradual series of steps.
