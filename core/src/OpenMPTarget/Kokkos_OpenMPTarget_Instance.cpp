#include <OpenMPTarget/Kokkos_OpenMPTarget_Instance.hpp>

#include <sstream>

namespace Kokkos {
namespace Experimental {
namespace Impl {
void OpenMPTargetInternal::fence() {}
int OpenMPTargetInternal::concurrency() { return 128000; }
const char* OpenMPTargetInternal::name() { return "OpenMPTarget"; }
void OpenMPTargetInternal::print_configuration(std::ostream& /*stream*/,
                                               const bool) {
  // FIXME_OPENMPTARGET
  printf("Using OpenMPTarget\n");
}

void OpenMPTargetInternal::impl_finalize() { m_is_initialized = false; }
void OpenMPTargetInternal::impl_initialize() { m_is_initialized = true; }
int OpenMPTargetInternal::impl_is_initialized() {
  return m_is_initialized ? 1 : 0;
}

OpenMPTargetInternal* OpenMPTargetInternal::impl_singleton() {
  static OpenMPTargetInternal self;
  return &self;
}

}  // Namespace Impl

OpenMPTarget::OpenMPTarget()
    : m_space_instance(Impl::OpenMPTargetInternal::impl_singleton()) {}

const char* OpenMPTarget::name() {
  return Impl::OpenMPTargetInternal::impl_singleton()->name();
}
void OpenMPTarget::print_configuration(std::ostream& stream,
                                       const bool detail) {
  m_space_instance->print_configuration(stream, detail);
}

int OpenMPTarget::concurrency() {
  return Impl::OpenMPTargetInternal::impl_singleton()->concurrency();
}
void OpenMPTarget::fence() {
  Impl::OpenMPTargetInternal::impl_singleton()->fence();
}

void OpenMPTarget::impl_initialize() { m_space_instance->impl_initialize(); }
void OpenMPTarget::impl_finalize() { m_space_instance->impl_finalize(); }
int OpenMPTarget::impl_is_initialized() {
  return Impl::OpenMPTargetInternal::impl_singleton()->impl_is_initialized();
}
}  // Namespace Experimental

namespace Impl {
int g_openmptarget_space_factory_initialized =
    Kokkos::Impl::initialize_space_factory<OpenMPTargetSpaceInitializer>(
        "160_OpenMPTarget");

void OpenMPTargetSpaceInitializer::initialize(const InitArguments& args) {
  // Prevent "unused variable" warning for 'args' input struct.  If
  // Serial::initialize() ever needs to take arguments from the input
  // struct, you may remove this line of code.
  (void)args;

  if (std::is_same<Kokkos::Experimental::OpenMPTarget,
                   Kokkos::DefaultExecutionSpace>::value) {
    Kokkos::Experimental::OpenMPTarget().impl_initialize();
    // std::cout << "Kokkos::initialize() fyi: OpenMP enabled and initialized"
    // << std::endl ;
  } else {
    // std::cout << "Kokkos::initialize() fyi: OpenMP enabled but not
    // initialized" << std::endl ;
  }
}

void OpenMPTargetSpaceInitializer::finalize(const bool all_spaces) {
  if (std::is_same<Kokkos::Experimental::OpenMPTarget,
                   Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::Experimental::OpenMPTarget().impl_is_initialized())
      Kokkos::Experimental::OpenMPTarget().impl_finalize();
  }
}

void OpenMPTargetSpaceInitializer::fence() {
  Kokkos::Experimental::OpenMPTarget::fence();
}

void OpenMPTargetSpaceInitializer::print_configuration(std::ostringstream& msg,
                                                       const bool detail) {
  msg << "OpenMPTarget Execution Space:" << std::endl;
  msg << "  KOKKOS_ENABLE_OPENMPTARGET: ";
  msg << "yes" << std::endl;

  msg << "\nOpenMPTarget Runtime Configuration:" << std::endl;
  Kokkos::Experimental::OpenMPTarget().print_configuration(msg, detail);
}

}  // namespace Impl
}  // Namespace Kokkos
