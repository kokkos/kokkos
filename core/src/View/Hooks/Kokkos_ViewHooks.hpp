#ifndef KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
#define KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP

#include <Kokkos_Core_fwd.hpp>
#include <functional>
#include <mutex>

namespace Kokkos {
namespace Experimental {

class ViewHolder;
class ConstViewHolder;

namespace Detail {
template <typename View>
using copy_subscription_function_type = void (*)(View &, const View &);

template <template <typename> class Invoker, typename... Subscribers>
struct invoke_subscriber_impl;

template <template <typename> class Invoker>
struct invoke_subscriber_impl<Invoker> {
  template <typename... Args>
  static void invoke(Args &&...) {}
};

template <template <typename> class Invoker, typename Subscriber,
          typename... RemSubscribers>
struct invoke_subscriber_impl<Invoker, Subscriber, RemSubscribers...> {
  template <typename... Args>
  static void invoke(Args &&... args) {
    Invoker<Subscriber>::call(std::forward<Args>(args)...);
    invoke_subscriber_impl<Invoker, RemSubscribers...>::invoke(
        std::forward<Args>(args)...);
  }
};

template <typename Subscriber>
struct copy_constructor_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::copy_constructed(self, other);
  }
};

template <typename Subscriber>
struct move_constructor_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::move_constructed(self, other);
  }
};

template <typename Subscriber>
struct copy_assign_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::copy_assigned(self, other);
  }
};

template <typename Subscriber>
struct move_assign_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::move_assigned(self, other);
  }
};
}  // namespace Detail

struct EmptyViewHooks {
  using hooks_policy = EmptyViewHooks;

  template <typename View>
  static void copy_construct(View &, const View &) {}
  template <typename View>
  static void copy_assign(View &, const View &) {}
  template <typename View>
  static void move_construct(View &, const View &) {}
  template <typename View>
  static void move_assign(View &, const View &) {}
};

/* Will be some sort of mixin, not with a flag but with a type */
/* Because that functionality is not available yet we'll go with this. */
template <class... Subscribers>
struct SubscribableViewHooks {
  using hooks_policy = SubscribableViewHooks<Subscribers...>;

  template <typename View>
  static void copy_construct(View &self, const View &other) {
    Detail::invoke_subscriber_impl<Detail::copy_constructor_invoker,
                                   Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void copy_assign(View &self, const View &other) {
    Detail::invoke_subscriber_impl<Detail::copy_assign_invoker,
                                   Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void move_construct(View &self, const View &other) {
    Detail::invoke_subscriber_impl<Detail::move_constructor_invoker,
                                   Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void move_assign(View &self, const View &other) {
    Detail::invoke_subscriber_impl<Detail::move_assign_invoker,
                                   Subscribers...>::invoke(self, other);
  }
};

namespace Impl
{
template <class ViewType, class Traits = typename ViewType::traits,
    class Enabled = void>
struct DynamicViewHooksCaller;
}

struct DynamicViewHooksSubscriber {
  template <typename View>
  static void copy_constructed(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_copy_construct_hooks(self);
  }
  template <typename View>
  static void copy_assigned(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_copy_assign_hooks(self);
  }

  template <typename View>
  static void move_constructed(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_move_construct_hooks(self);
  }
  template <typename View>
  static void move_assigned(View &self, const View &) {
    Impl::DynamicViewHooksCaller<View>::call_move_assign_hooks(self);
  }
};

#ifdef KOKKOS_ENABLE_EXPERIMENTAL_DEFAULT_DYNAMIC_VIEWHOOKS
using DefaultViewHooks = SubscribableViewHooks< DynamicViewHooksSubscriber >;
#else
using DefaultViewHooks = EmptyViewHooks;
#endif

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_EXPERIMENTAL_ACCESSORHOOKS_HPP
