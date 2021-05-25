#ifndef KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
#define KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP

#include <functional>
#include <Kokkos_ViewHolder.hpp>

namespace Kokkos
{
namespace Experimental
{

namespace Detail
{
template< typename View >
using copy_subscription_function_type = void (*)( View &, const View & );

template< template< typename > class Invoker, typename... Subscribers >
struct invoke_subscriber_impl;

template< template< typename > class Invoker >
struct invoke_subscriber_impl< Invoker >
{
  template< typename... Args >
  static void invoke( Args &&... ) {}
};

template< template< typename > class Invoker, typename Subscriber, typename... RemSubscribers >
struct invoke_subscriber_impl< Invoker, Subscriber, RemSubscribers... >
{
  template< typename... Args >
  static void invoke( Args &&..._args )
  {
    Invoker< Subscriber >::call( std::forward< Args >( _args )... );
    invoke_subscriber_impl< Invoker, RemSubscribers... >::invoke( std::forward< Args >( _args )... );
  }
};

template< typename Subscriber >
struct copy_constructor_invoker
{
  template< typename View >
  static void call( View &self, const View &other )
  {
    Subscriber::copy_constructed( self, other );
  }
};
}

struct DefaultViewHooks
{
  using hooks_policy = DefaultViewHooks;

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
  using hooks_policy = SubscribableViewHooks< Subscribers... >;

  template< typename View >
  static void construct( View &view ) {}
  template< typename View >
  static void copy_construct( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::copy_constructor_invoker, Subscribers... >::invoke( self, other );
  }
  template< typename View >
  static void copy_assign( View &view ) {}
  template< typename View >
  static void move_construct( View &view ) {}
  template< typename View >
  static void move_assign( View &view ) {}
};


struct DynamicViewHooks {
  using callback_type       = std::function<void(ViewHolderBase &)>;
  using const_callback_type = std::function<void(ConstViewHolderBase &)>;

  template <typename F, typename ConstF>
  static void set(F &&fun, ConstF &&const_fun) {
    s_callback       = std::forward<F>(fun);
    s_const_callback = std::forward<ConstF>(const_fun);
  }

  static void clear() {
    s_callback       = callback_type{};
    s_const_callback = const_callback_type{};
  }

  static bool is_set() noexcept {
    return static_cast<bool>(s_callback) || static_cast<bool>(s_const_callback);
  }

  template <class DataType, class... Properties>
  static void call(View<DataType, Properties...> &view) {
    callback_type tmp_callback;
    const_callback_type tmp_const_callback;

    std::swap(s_callback, tmp_callback);
    std::swap(s_const_callback, tmp_const_callback);

    auto holder = ViewHolder<View<DataType, Properties...> >(view);

    do_call(tmp_callback, tmp_const_callback, std::move(holder));

    std::swap(s_callback, tmp_callback);
    std::swap(s_const_callback, tmp_const_callback);
  }

 private:
  static void do_call(callback_type _cb, const_callback_type _ccb,
                      ViewHolderBase &&view) {
    if (_cb) {
      _cb(view);
    }
  }

  static void do_call(callback_type _cb, const_callback_type _ccb,
                      ConstViewHolderBase &&view) {
    if (_ccb) _ccb(view);
  }

  static callback_type s_callback;
  static const_callback_type s_const_callback;
};


namespace Impl {
template <class ViewType, class Traits = typename ViewType::traits,
    class Enabled = void>
struct DynamicViewHooksCaller {
  static void call(ViewType &view) {}
};

template <class ViewType, class Traits>
struct DynamicViewHooksCaller<
    ViewType, Traits,
    typename std::enable_if<!std::is_same<typename Traits::memory_space,
        AnonymousSpace>::value>::type> {
  static void call(ViewType &view) {
    if (DynamicViewHooks::is_set()) DynamicViewHooks::call(view);
  }
};
}  // namespace Impl

struct DynamicViewHooksSubscriber
{
  template< typename View >
  static void copy_constructed( View &self, const View & )
  {
    Impl::DynamicViewHooksCaller< View >::call( self );
  }
};
} // namespace Experimental
} // namespace Kokkos

#endif  // KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
