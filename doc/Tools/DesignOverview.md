# Design Overview

This document covers the high-level design goals and implementation of the Kokkos Tools subsystem.

## History

Much of the design of Kokkos Tools has influences from its history. As Kokkos was being developed, the developers wanted ways to check the performance of Kokkos code in a platform and backend-agnostic way. Suppose Christian had some clever new optimization in Kokkos itself, and wanted to check whether this was (a) brilliant or (b) deeply stupid. As this capability was developed, it became more and more clear that these facilities were useful to developers of Kokkos applications. 

## Goals

One of the early goals of the Kokkos Tools subsystem was to make it really easy to develop simple tools for simple tasks. Specifically, it was seen as deeply obnoxious to require

- A tool to need some version-dependent Kokkos_Make_The_Tool_Work.hpp file to achieve simple goals
- An app developer to recompile their code to change or add a tool
- Any noticeable overhead when a tool was not in use

The first is for the convenience of tool developers, and it is critically important. As of today, we _do_ have header files for some advanced Kokkos Tools functionality in the Kokkos_Profiling_C_Interface.h and Kokkos_Profiling.hpp files, but these are backwards compatible for almost five years. If suddenly TAU needs to know whether it's being used by Kokkos 3.6 or Kokkos 3.6.1, we will have failed in a promise to tool developers. This matters because right now, Tools have to guard profiling layers like OMPT behind an "ENABLE_OMPT" build config option, which defaults to off. Meanwhile, an increasing set of tools either have no build option to disable Kokkos Tools (see: TAU), or at least to default to building Kokkos Tools (see: Caliper).

## A Tool's Perspective: What do Tools Get From Kokkos, What Comes from Other Sources?

Request from Bill Williams, currently a stub section