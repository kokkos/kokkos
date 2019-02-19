#ifndef hostspace_openmp_deepcopy_HPP
#define hostspace_openmp_deepcopy_HPP

#include <Kokkos_Macros.hpp>

#if defined( KOKKOS_ENABLE_OPENMP )

#include <cstring> // size_t

namespace Kokkos {

namespace Impl {

void hostspace_openmp_deepcopy(void * dst, const void * src, size_t n);

} // namespace Impl

} // namespace Kokkos


// this symbol should not exist without openmp, a different threading model
// would require a slightly different impl
#endif // if defined( KOKKOS_ENABLE_OPENMP )

#endif // hostspace_openmp_deepcopy_HPP
