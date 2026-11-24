``ErrorReporter``
=================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_ErrorReporter.hpp>``

``ErrorReporter`` is a class that can collect error reports in a thread safe manner.
The report type is user defined, and it will only store errors up to a defined capacity.

Interface
---------

.. cpp:class:: template<class ReportType, class DeviceType> ErrorReporter

   A class to collect error reports in a thread-safe manner.

   :tparam ReportType: The type of the reported error data. Must be a valid element type for :cpp:struct:`View`.

   :tparam DeviceType: The device type of the ``ErrorReporter``. Default is ``DefaultExecutionSpace::device_type``. 

   |

   .. rubric:: *Public* typedefs

   .. cpp:type:: report_type

      :cpp:any:`ReportType` This is the type for the stored error data, which can be user defined.

   .. cpp:type:: device_type

      :cpp:any:`DeviceType` The device type defining in which execution space reports can be added.

   .. cpp:type:: execution_space

      :cpp:type:`device_type::execution_space`


   .. rubric:: *Public* constructors

   .. cpp:function:: ErrorReporter(const std::string& label, int size)

      Constructs a new ErrorReporter instance with capacity of size.
   
   .. cpp:function:: ErrorReporter(int size)

      Constructs a new ErrorReporter instance with capacity of size.

   .. rubric:: member functions

   .. cpp:function:: int capacity() const

      :returns: The maximum number of errors the instance can store.
   
   .. cpp:function:: int num_reports() const

      :returns: The number of errors that were recorded.

   .. cpp:function:: int num_report_attempts() const

      :returns: The number of errors that were attempted to be recorded. Equal to :cpp:any:`num_reports` if the number of attempts was less than :cpp:any:`capacity`.

   .. cpp:function:: std::pair<std::vector<int>, std::vector<report_type>> get_reports() const

      :returns: Two ``std::vector`` containing the ids of the reporters, and the reports themselves. The size of the vectors is equal to :cpp:any:`num_reports()`.

   .. cpp:function:: bool full() const

      :returns: ``true`` if and only if the number of attempted reports is equal or exceeds :cpp:any:`capacity()`.

   .. cpp:function:: void clear() const

      Resets the error reporter. :cpp:any:`num_reports()` is zero after this operation, and new errors can be recorded.

   .. cpp:function:: void resize(const size_t size)

      Changes the capacity of the instance to ``size``. Existing error reports may be lost.

   .. cpp:function:: bool add_report(int reporter_id, report_type report) const
      
      Attempts to record an error. If space is available ``report`` is stored and the attempt is successful.

      :returns: ``true`` if and only if the attempt to record the error was successful.


Example
-------

.. code-block:: cpp
  
   #include <Kokkos_Core.hpp>
   #include <Kokkos_ErrorReporter.hpp>
   #include <Kokkos_Random.hpp>

   // Kokkos ErrorReporter can be used to record a certain
   // number of errors up to a point for debugging purposes.
   // The main benefit of ErrorReporter is that its thread safe
   // and does not require storage that depends on the concurrency
   // of the architecture you are running on.

   // This little example assumes you want to sort particles
   // based on their position into boxes, but it will report
   // if any of the particles are outside of the boxes.
   int main(int argc, char* argv[]) {
     Kokkos::initialize(argc, argv);
     {
       Kokkos::View<double*> positions("Pos", 10000);
       Kokkos::View<int*> box_id("box_id");

       // Lets produce some random positions in the range of -5 to 105
       Kokkos::Random_XorShift64_Pool<> rand_pool(103201);
       Kokkos::fill_random(positions, rand_pool, -5., 105.);

       // Now create an error reporter that can store 10 reports
       // We will simply report the position, but it could be a user
       // defined type.
       Kokkos::Experimental::ErrorReporter<double> errors("MyErrors", 10);

       // Counting how many positions fall into the 0-50 and 50-100 range
       int num_lower_box = 0;
       int num_upper_box = 0;
       Kokkos::parallel_reduce(
           "ErrorReporter Example", positions.extent(0),
           KOKKOS_LAMBDA(int i, int& count_lower, int& count_upper) {
             double pos = positions(i);
             // Check for positions outside the range first
             if (pos < 0. || pos > 100.) {
               // add_report takes an id and a payload
               // Note that we don't have to check how many reports were already
               // submitted
               errors.add_report(i, pos);
             } else if (pos < 50.)
               count_lower++;
             else
               count_upper++;
           },
           num_lower_box, num_upper_box);

       // Lets report results
       printf(
           "Of %i particles %i fall into the lower box, and %i into the upper "
           "box\n",
           positions.extent_int(0), num_lower_box, num_upper_box);

       // Lets report errors
       printf(
           "There were %i particles outside of the valid domain (0 - 100). Here "
           "are the first %i:\n",
           errors.num_report_attempts(), errors.num_reports());

       // Using structured bindings to get the reporter ids and reports
       auto [reporter_ids, reports] = errors.get_reports();
       for (int e = 0; e < errors.num_reports(); e++)
         printf("%i %lf\n", reporter_ids[e], reports[e]);
     }
     Kokkos::finalize();
   }
