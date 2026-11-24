.. role:: cpp(code)
    :language: cpp

Multi-GPU Support
=================

Kokkos has support for launching kernels on multiple GPUs from a single host process, e.g., a single MPI rank. This feature currently
exists for the CUDA, HIP, and SYCL backends.

Using this feature requires knowledge of backend specific API calls for creating non-default execution space instances. Once the execution space has been
created, it can be used to create execution policies, allocate views, etc. all on the device chosen by the user.

Constructing Execution Spaces
-----------------------------

CUDA
~~~~

For CUDA backend, the user creates a ``cudaStream_t`` object and passes it to the ``Kokkos::Cuda`` constructor.

.. note:: The lifetime of all Kokkos objects related to the stream (e.g., the execution space) needs to end before the stream itself is invalidated
          via ``cudaStreamDestroy``. This can be done by adding a scope around such objects, as is done below.


.. code-block:: cpp

    // Query number of devices available
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    // Choose some 0 <= N < n_devices
    int N = ...;

    // Create stream on device N
    cudaStream_t stream;
    cudaSetDevice(N);
    cudaStreamCreate(&stream);

    // Scope execution space to ensure stream
    // is destroyed *after* execution space
    {
      // Create execution space
      Kokkos::Cuda exec_space(stream);

      // Use execution space
      /* ... */
    }

    // Destroying stream after use
    cudaSetDevice(N);
    cudaStreamDestroy(stream);

HIP
~~~

For the HIP backend, like with CUDA, the user creates a ``hipStream_t`` object and passes it to the ``Kokkos::HIP`` constructor.

.. note:: The lifetime of all Kokkos objects related to the stream (e.g., the execution space) needs to end before the stream itself is invalidated
          via ``hipStreamDestroy``. This can be done by adding a scope around such objects, as is done below.

.. warning:: Multi-GPU only supported for ROCm 5.6 and later. Because of the lack of HIP API functions for querying a
             stream's device before ROCm 5.6, constructing a ``Kokkos::HIP`` instance on a non-default device isn't
             supported.


.. code-block:: cpp

    // Query number of devices available
    int n_devices;
    hipGetDeviceCount(&n_devices);

    // Choose some 0 <= N < n_devices
    int N = ...;

    // Create stream on device N
    hipStream_t stream;
    hipSetDevice(N);
    hipStreamCreate(&stream);

    // Scope execution space to ensure stream
    // is destroyed *after* execution space
    {
      // Create execution space
      Kokkos::HIP exec_space(stream);

      // Use execution space
      /* ... */
    }

    // Destroying stream after use
    hipSetDevice(N);
    hipStreamDestroy(stream);

SYCL
~~~~

For the SYCL backend, the user creates a ``sycl::queue`` object and passes it to the ``Kokkos::SYCL`` constructor.

.. code-block:: cpp

    // Get list of devices available
    std::vector<sycl::device> gpu_devices =
      sycl::device::get_devices(sycl::info::device_type::gpu);

    // Choose some 0 <= N < gpu_devices.size()
    int N = ...;

    // Create a queue on device N.
    // Note: Kokkos requires SYCL queues to be "in_order"
    sycl::queue queue{gpu_devices[N], sycl::property::queue::in_order()};

    // Create execution space
    Kokkos::SYCL exec_space(queue);

    // Use execution space
    /* ... */

Using Kokkos Methods
--------------------

Once an execution space has been created on the chosen device, the execution space must be passed to all Kokkos methods
intended to be used on the chosen device. If no execution space is passed, Kokkos will use the default execution space
instance associated with the device with which Kokkos was initialized.

Allocating Managed Views
~~~~~~~~~~~~~~~~~~~~~~~~

To allocate a managed view on device, pass the execution space to ``Kokkos::view_alloc()``.

Example:

.. code-block:: cpp

    using ExecutionSpace = decltype(exec_space);
    Kokkos::View<int*, typename ExecutionSpace::memory_space> V(Kokkos::view_alloc("V", exec_space), 10);

Launching Kernels
~~~~~~~~~~~~~~~~~

To launch a kernel on device, pass the execution space to the policy constructor.

Example:

.. code-block:: cpp

    Kokkos::parallel_for("inc_V", Kokkos::RangePolicy(exec_space, 0, 10),
      KOKKOS_LAMBDA (const int i) {
        V(i) += i;
    });

Notes
-----

- A `tutorial <https://github.com/kokkos/kokkos-tutorials/tree/main/Exercises/multi_gpu_cuda>`_ for using multi-GPU on CUDA is available.
