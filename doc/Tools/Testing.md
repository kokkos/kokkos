# Testing Subsubsystem for Tools

## Why?

One huge problem has been how to know whether the Kokkos Tools interface is correctly invoking events. In the dark, uncivilized ages (at time of writing: until yesterday) we used a tool that printed when it hit events, and regular expressions to parse the output. This is of really limited utility, brittle, and bound by your skill at regular expressions. The team decided we needed something better.

## Design

So, we need the ability to determine whether a given piece of code makes a correct set of events. That means that our interface needs a way to specify a piece of code, and a way of specifying an expected set of events.

This being Kokkos, the way you specify a piece of code is a lambda. But how do we specify a set of events to match? In the design, we wanted the ability to do more than an exact match. Some way to specify "hey, there's a begin_parallel_for event on a certain device ID, then a begin_fence_event on that same device ID."

The design uses lambdas for this. Essentially, there's a whole set of structs representing different events. Along with the lambda representing the code section, you can pass any number of lambdas, containing any number of these structs as argument types. You can then match whatever you want in fields of those structs (making sure device ID's match, or that names are good). Each event from the code section is matched against an argument in the lambda. If the wrong amounts of events are encountered, or events don't match, the checker fails.

Additionally, there are two more forms of matchers. One checks for the absence of a given event, that is, it pipes every event through every matcher, and if any _do_ match, the test fails. The final one simple returns a vector containing the captured events, if none of the lambda methods appeal, this allows a user to do whatever they want with the returned vector.

In this document, we actually won't document the implementation, the implementation itself is highly commented.