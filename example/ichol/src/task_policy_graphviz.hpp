#pragma once
#ifndef __TASK_POLICY_GRAPHVIZ_HPP__
#define __TASK_POLICY_GRAPHVIZ_HPP__

/// \file task_policy_graphviz.hpp
/// \brief Minimum implementation to mimic Kokkos task policy and future mechanism with graphviz output.
/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace Tacho { 

  using namespace std;

  static vector<string> g_graphviz_color;

  class Task : public Disp  {
  private:
    string _label;
    set<Task*> _dep_tasks;
    int _phase;

  public:
    Task() : _label("no-name"), _phase(0) { }
    Task(const Task &b) : _label(b._label), _phase(b._phase) { }
    Task(const string label, const int phase) : _label(label), _phase(phase) { }
    virtual~Task() { }

    void addDependence(Task *b) {  if (b != NULL) _dep_tasks.insert(b); }
    void clearDependence() { _dep_tasks.clear(); }
    
    ostream& showMe(ostream &os) const {
      os << "    uid = " << this 
         << " , label = " << _label 
         << ", # of deps = " << _dep_tasks.size()  
         << endl;                  
      if (_dep_tasks.size()) {
        for (auto it=_dep_tasks.begin();it!=_dep_tasks.end();++it)
          os << "          " << (*it) << " , name = " << (*it)->_label << endl;
      }
      return os;
    }
    
    ostream& graphviz(ostream &os, const size_t cnt) const {
      if (g_graphviz_color.size() == 0) {
        g_graphviz_color.push_back("indianred2");
        g_graphviz_color.push_back("lightblue2");
        g_graphviz_color.push_back("skyblue2");
        g_graphviz_color.push_back("lightgoldenrod2");
        g_graphviz_color.push_back("orange2");
        g_graphviz_color.push_back("mistyrose2");
      }

      // os << (long)(this)
      //    << " [label=\"" << cnt << " "<< _label;
      os << (long)(this)
         << " [label=\"" << _label;
      if (_phase > 0)
        os << "\" ,style=filled,color=\"" << g_graphviz_color.at(_phase%g_graphviz_color.size()) << "\" ];";
      else 
        os << "\"];";
      
      for (auto it=_dep_tasks.begin();it!=_dep_tasks.end();++it)
        os << (long)(*it) << " -> " << (long)this << ";";
      
      return (os << endl);
    }    
  };

  class Future {
  private:
    Task *_task;
    
  public:
    Future() : _task(NULL) { }
    Future(Task *task) : _task(task) { }
    Task* TaskPtr() const { return _task; }

  };

  class TeamThreadMember {
  public:
    TeamThreadMember() { }
    int team_rank() const { return 0; }
    int team_size() const { return 1; }
    void team_barrier() const { }
  };

  class TaskPolicy : public Disp {
  private:
    vector<Task*> _queue;
    int _work_phase;

  public:
    TaskPolicy() 
      : _queue(), 
        _work_phase(0) 
    { }

    // Kokkos interface
    // --------------------------------
    typedef class TeamThreadMember member_type;
    static member_type member_single() { return member_type(); }

    template<typename TaskFunctorType> 
    Future create(const TaskFunctorType &func, const int dep_size) {
      return Future(new Task(func.Label(), _work_phase));
    }

    template<typename TaskFunctorType> 
    Future create_team(const TaskFunctorType &func, const int dep_size) {
      return Future(new Task(func.Label(), _work_phase));
    }
    
    void spawn(const Future &obj) {
      _queue.push_back(obj.TaskPtr());
    }

    void add_dependence(const Future &after, const Future &before) {
      if (after.TaskPtr() != NULL)
        after.TaskPtr()->addDependence(before.TaskPtr());
    }
    
    void wait(const Future &obj) {
      // do nothing
    }

    // Graphviz interface
    // --------------------------------
    size_t size() const {
      return _queue.size();
    }

    void clear() {
      for (auto it=_queue.begin();it!=_queue.end();++it)
        delete (*it);

      _queue.clear();
    }

    void set_work_phase(const int phase) {
      _work_phase = phase;
    }

    int get_work_phase() const {
      return _work_phase;
    }
    
    ostream& showMe(ostream &os) const {
      if (_queue.size()) {
        os << " -- Task Queue -- " << endl;
        for (auto it=_queue.begin();it!=_queue.end();++it)
          (*it)->showMe(os);
      } else {
        os << " -- Task Queue is empty -- " << endl;
      }
      return os;
    }
    
    ostream& graphviz(ostream &os,
                      const double width = 7.5,
                      const double length = 10.0) {
      os << "digraph TaskGraph {" << endl;
      os << "size=\"" << width << "," << length << "\";" << endl;
      size_t count = 0;
      for (auto it=_queue.begin();it!=_queue.end();++it)
        (*it)->graphviz(os,count++);
      os << "}" << endl;
      os << "# total number of tasks = " << count << endl;
      
      return (os << endl);
    }
  };

}

#endif
