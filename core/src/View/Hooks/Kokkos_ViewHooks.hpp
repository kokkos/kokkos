#ifndef KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
#define KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP

#include <functional>
#include <mutex>
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

template< typename Subscriber >
struct move_constructor_invoker
{
  template< typename View >
  static void call( View &self, const View &other )
  {
    Subscriber::move_constructed( self, other );
  }
};

template< typename Subscriber >
struct copy_assign_invoker
{
  template< typename View >
  static void call( View &self, const View &other )
  {
    Subscriber::copy_assigned( self, other );
  }
};

template< typename Subscriber >
struct move_assign_invoker
{
  template< typename View >
  static void call( View &self, const View &other )
  {
    Subscriber::move_assigned( self, other );
  }
};
}

struct DefaultViewHooks
{
  using hooks_policy = DefaultViewHooks;

  template< typename View >
  static void copy_construct( View &, const View & ) {}
  template< typename View >
  static void copy_assign( View &, const View & ) {}
  template< typename View >
  static void move_construct( View &, const View & ) {}
  template< typename View >
  static void move_assign( View &, const View & ) {}
};

/* Will be some sort of mixin, not with a flag but with a type */
/* Because that functionality is not available yet we'll go with this. */
template< class... Subscribers >
struct SubscribableViewHooks
{
  using hooks_policy = SubscribableViewHooks< Subscribers... >;

  template< typename View >
  static void copy_construct( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::copy_constructor_invoker, Subscribers... >::invoke( self, other );
  }
  template< typename View >
  static void copy_assign( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::copy_assign_invoker, Subscribers... >::invoke( self, other );
  }
  template< typename View >
  static void move_construct( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::move_constructor_invoker, Subscribers... >::invoke( self, other );
  }
  template< typename View >
  static void move_assign( View &self, const View &other ) {
    Detail::invoke_subscriber_impl< Detail::move_assign_invoker, Subscribers... >::invoke( self, other );
  }
};


struct DynamicViewHooks {
  using callback_type       = std::function<void(const ViewHolder &)>;
  using const_callback_type = std::function<void(const ConstViewHolder &)>;

  class callback_overload_set
  {
   public:

    template <class DataType, class... Properties>
    void call(View<DataType, Properties...> &view) {
      std::lock_guard< std::mutex > lock( m_mutex );

      if ( !any_set() )
        return;

      auto holder = make_view_holder(view);
      do_call(std::move(holder));
    }

    template< typename F >
    void set_callback( F &&cb )
    {
      std::lock_guard< std::mutex > lock( m_mutex );
      m_callback = std::forward< F >( cb );
    }

    template< typename F >
    void set_const_callback( F &&cb )
    {
      std::lock_guard< std::mutex > lock( m_mutex );
      m_const_callback = std::forward< F >( cb );
    }

    void clear_callback()
    {
      std::lock_guard< std::mutex > lock( m_mutex );
      m_callback = {};
    }

    void clear_const_callback()
    {
      std::lock_guard< std::mutex > lock( m_mutex );
      m_const_callback = {};
    }

    void reset()
    {
      std::lock_guard< std::mutex > lock( m_mutex );
      m_callback = {};
      m_const_callback = {};
    }

   private:

    // Not thread safe, don't call outside of mutex lock
    void do_call(const ViewHolder &view) {
      if (m_callback) m_callback(view);
    }

    // Not thread safe, don't call outside of mutex lock
    void do_call(const ConstViewHolder &view) {
      if (m_const_callback) m_const_callback(view);
    }

    // Not thread safe, call inside of mutex
    bool any_set() const noexcept
    {
      return static_cast< bool >( m_callback ) || static_cast< bool >( m_const_callback );
    }

    callback_type m_callback;
    const_callback_type m_const_callback;
    std::mutex m_mutex;
  };

  static void reset()
  {
    copy_constructor_set.reset();
    copy_assignment_set.reset();
    move_constructor_set.reset();
    move_assignment_set.reset();
  }

  static callback_overload_set copy_constructor_set;
  static callback_overload_set copy_assignment_set;
  static callback_overload_set move_constructor_set;
  static callback_overload_set move_assignment_set;
  static thread_local bool reentrant; // don't enter *any* callback while in a callback on this thread
};


namespace Impl {
template <class ViewType, class Traits = typename ViewType::traits,
    class Enabled = void>
struct DynamicViewHooksCaller {
  static void call_construct_hooks(ViewType &) {}
  static void call_copy_construct_hooks(ViewType &) {}
  static void call_copy_assign_hooks(ViewType &) {}
  static void call_move_construct_hooks(ViewType &) {}
  static void call_move_assign_hooks(ViewType &) {}
};

template <class ViewType, class Traits>
struct DynamicViewHooksCaller<
    ViewType, Traits,
    typename std::enable_if<!std::is_same<typename Traits::memory_space,
        AnonymousSpace>::value>::type> {
  static void call_copy_construct_hooks(ViewType &view) {
    static thread_local bool reentrant = false;
    if ( !reentrant ) {
      reentrant = true;
      DynamicViewHooks::copy_constructor_set.call(view);
      reentrant = false;
    }
  }

  static void call_copy_assign_hooks(ViewType &view) {
    if ( !DynamicViewHooks::reentrant ) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::copy_assignment_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }

  static void call_move_construct_hooks(ViewType &view) {
    if ( !DynamicViewHooks::reentrant ) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::move_constructor_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }

  static void call_move_assign_hooks(ViewType &view) {
    if ( !DynamicViewHooks::reentrant ) {
      DynamicViewHooks::reentrant = true;
      DynamicViewHooks::move_assignment_set.call(view);
      DynamicViewHooks::reentrant = false;
    }
  }
};
}  // namespace Impl

struct DynamicViewHooksSubscriber
{
  template< typename View >
  static void copy_constructed( View &self, const View &other ) {
    Impl::DynamicViewHooksCaller< View >::call_copy_construct_hooks( self );
  }
  template< typename View >
  static void copy_assigned( View &self, const View &other ) {
    Impl::DynamicViewHooksCaller< View >::call_copy_assign_hooks( self );
  }

  template< typename View >
  static void move_constructed( View &self, const View &other ) {
    Impl::DynamicViewHooksCaller< View >::call_move_construct_hooks( self );
  }
  template< typename View >
  static void move_assigned( View &self, const View &other ) {
    Impl::DynamicViewHooksCaller< View >::call_move_assign_hooks( self );
  }
};
} // namespace Experimental
} // namespace Kokkos

#endif  // KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
