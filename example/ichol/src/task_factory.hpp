#pragma once
#ifndef __TASK_FACTORY_HPP__
#define __TASK_FACTORY_HPP__

/// \file task_factory.hpp
/// \brief A wrapper for task policy and future with a provided space type.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho { 

  using namespace std;

  /// \class TaskFactory
  /// \brief Minimal interface to Kokkos tasking.
  ///
  /// TaskFactory is attached to blocks as a template argument in order to 
  /// create and manage tasking future objects. Note that policy (shared 
  /// pointer to the task generator) is not a member object in this class.
  /// This class includes minimum interface for tasking with type decralation 
  /// of the task policy and template alias of future so that future objects 
  /// generated in this class will match to their policy and its execution space. 
  ///
  template<typename PolicyType,        
           typename FutureType>
  class TaskFactory {
  private:
    static PolicyType *_policy;
    static int _max_task_dependence;
    static bool _use_team_interface;

  public:
    typedef PolicyType policy_type;
    typedef FutureType future_type;
    
    template<typename TaskFunctorType>
    static 
    future_type create(policy_type &policy, const TaskFunctorType &func) {
      return (_use_team_interface ? 
              policy.task_create_team(func, _max_task_dependence) : 
              policy.task_create     (func, _max_task_dependence)); 
    }
    
    static
    void spawn(policy_type &policy, const future_type &obj) {
      policy.spawn(obj);
    }
    
    static
    void addDependence(policy_type &policy, 
                       const future_type &after, const future_type &before) {
      policy.add_dependence(after, before);
    }

    template<typename TaskFunctorType>
    static 
    void addDependence(policy_type &policy, 
                       TaskFunctorType *after, const future_type &before) {
      policy.add_dependence(after, before);
    }

    template<typename TaskFunctorType>
    static 
    void clearDependence(policy_type &policy, TaskFunctorType *func) {
      policy.clear_dependence(func);
    }

    template<typename TaskFunctorType>
    static
    void respawn(policy_type &policy, TaskFunctorType *func) {
      policy.respawn(func);
    }

    static
    void setPolicy(policy_type *policy) {
      _policy = policy;
    }

    static 
    void setUseTeamInterface(const bool use_team_interface) {
      _use_team_interface = use_team_interface;
    }

    static 
    void setMaxTaskDependence(const int max_task_dependence) {
      _max_task_dependence = max_task_dependence;
    }

    static
    policy_type& Policy() { 
      return *_policy;
    } 

  };

  template<typename PolicyType, typename FutureType> 
  PolicyType* TaskFactory<PolicyType,FutureType>::_policy = NULL;

  template<typename PolicyType, typename FutureType> 
  bool TaskFactory<PolicyType,FutureType>::_use_team_interface = true;

  template<typename PolicyType, typename FutureType> 
  int TaskFactory<PolicyType,FutureType>::_max_task_dependence = 10;
}

#endif
