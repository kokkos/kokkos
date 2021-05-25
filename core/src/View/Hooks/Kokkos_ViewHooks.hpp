#ifndef KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
#define KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP

#include <functional>

namespace Kokkos
{
namespace Experimental
{

namespace Detail
{
template< typename View >
using copy_subscription_function_type = void (*)( View &, const View & );

template< typename FunType, FunType *...Rems >
struct invoke_subscriber_impl;

template< typename FunType >
struct invoke_subscriber_impl< FunType >
{
  template< typename... Args >
  static void invoke( Args &&... ) {}
};

template< typename FunType, FunType *First, FunType *...Rems >
struct invoke_subscriber_impl< FunType, First, Rems... >
{
  template< typename... Args >
  static void invoke( Args &&..._args )
  {
    First( std::forward< Args >( _args )... );
    invoke_subscriber_impl< FunType, Rems... >::invoke( std::forward< Args >( _args )... );
  }
};
}

struct DefaultViewHooks
{
  template< typename View >
  static void construct( View &view ) {}
  template< typename View >
  static void copy_construct( View &self, const View &other ) {}
  template< typename View >
  static void copy_assign( View &view ) {}
  template< typename View >
  static void move_construct( View &view ) {}
  template< typename View >
  static void move_assign( View &view ) {}
};

/* Will be some sort of mixin, not with a flag but with a type */
/* Because that functionality is not available yet we'll go with this. */
template< class... Subscribers >
struct SubscribableViewHooks
{
  template< typename View >
  static void construct( View &view ) {}
  template< typename View >
  static void copy_construct( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::copy_subscription_function_type< View >, &Subscribers::copy_constructed... >::invoke( self, other );
  }
  template< typename View >
  static void copy_assign( View &view ) {}
  template< typename View >
  static void move_construct( View &view ) {}
  template< typename View >
  static void move_assign( View &view ) {}
};
}
}

#endif  // KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
