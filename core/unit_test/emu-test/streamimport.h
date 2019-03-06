#ifndef __KOKKOS_STREAM_IMPORT_
#define __KOKKOS_STREAM_IMPORT_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>

class kokkosstream {

private:
   std::string myString;
public:
   kokkosstream();
   kokkosstream& operator<<(const std::string& str);
   kokkosstream& operator<<(const char * st);
   kokkosstream& operator<<(bool __n);
   kokkosstream& operator<<(short __n);
   kokkosstream& operator<<(unsigned short __n);
   kokkosstream& operator<<(int __n);
   kokkosstream& operator<<(unsigned int __n);
   kokkosstream& operator<<(long __n);
   kokkosstream& operator<<(unsigned long __n);
   kokkosstream& operator<<(long long __n);
   kokkosstream& operator<<(unsigned long long __n);
   kokkosstream& operator<<(float __f);
   kokkosstream& operator<<(double __f);
   kokkosstream& operator<<(long double __f);
   
   std::string& str() {
      return myString;
   }

};

namespace std {
   typedef kokkosstream stringstream;
}


#endif
