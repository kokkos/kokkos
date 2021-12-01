#ifndef KOKKOS_EXPERIMENTAL_DYNAMICVIEWHOOKS_HPP
#define KOKKOS_EXPERIMENTAL_DYNAMICVIEWHOOKS_HPP

#include <Kokkos_ViewHolder.hpp>

namespace Kokkos
{
namespace Experimental
{
struct DynamicViewHooks {
  using callback_type       = std::function<void(const ViewHolder &)>;
  using const_callback_type = std::function<void(const ConstViewHolder &)>;

  class callback_overload_set {
   public:
    template <class DataType, class... Properties>
        void call(View<DataType, Properties...> &view) {
          std::lock_guard<std::mutex> lock(m_mutex);

          if (!any_set()) return;

          auto holder = make_view_holder(view);
          do_call(std::move(holder));
        }

        template <typename F>
        void set_callback(F &&cb) {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_callback = std::forward<F>(cb);
        }

        template <typename F>
        void set_const_callback(F &&cb) {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_const_callback = std::forward<F>(cb);
        }

        void clear_callback() {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_callback = {};
        }

        void clear_const_callback() {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_const_callback = {};
        }

        void reset() {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_callback       = {};
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
    bool any_set() const noexcept {
      return static_cast<bool>(m_callback) ||
      static_cast<bool>(m_const_callback);
    }

    callback_type m_callback;
    const_callback_type m_const_callback;
    std::mutex m_mutex;
  };

  static void reset() {
    copy_constructor_set.reset();
    copy_assignment_set.reset();
    move_constructor_set.reset();
    move_assignment_set.reset();
  }

  static callback_overload_set copy_constructor_set;
  static callback_overload_set copy_assignment_set;
  static callback_overload_set move_constructor_set;
  static callback_overload_set move_assignment_set;
  static thread_local bool reentrant;  // don't enter *any* callback while in a
  // callback on this thread
};

namespace Impl {
template <class ViewType, class Traits,
    class Enabled >
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
        if (!reentrant) {
          reentrant = true;
          DynamicViewHooks::copy_constructor_set.call(view);
          reentrant = false;
        }
      }

      static void call_copy_assign_hooks(ViewType &view) {
        if (!DynamicViewHooks::reentrant) {
          DynamicViewHooks::reentrant = true;
          DynamicViewHooks::copy_assignment_set.call(view);
          DynamicViewHooks::reentrant = false;
        }
      }

      static void call_move_construct_hooks(ViewType &view) {
        if (!DynamicViewHooks::reentrant) {
          DynamicViewHooks::reentrant = true;
          DynamicViewHooks::move_constructor_set.call(view);
          DynamicViewHooks::reentrant = false;
        }
      }

      static void call_move_assign_hooks(ViewType &view) {
        if (!DynamicViewHooks::reentrant) {
          DynamicViewHooks::reentrant = true;
          DynamicViewHooks::move_assignment_set.call(view);
          DynamicViewHooks::reentrant = false;
        }
      }
    };
}  // namespace Impl
}
}

#endif  // KOKKOS_EXPERIMENTAL_DYNAMICVIEWHOOKS_HPP
