#include "streamimport.h"
#include <ostream>
#include <sstream>

kokkosstream::kokkosstream() {

}

kokkosstream& kokkosstream::operator <<(const std::string& str) {
   size_t nLen = myString.length() + str.length();
   char * buf = (char*)malloc((nLen+1) * sizeof(char));
   if (myString.length() > 0)
      memcpy(&buf[0], myString.c_str(), myString.length());
   if (str.length() > 0)
      memcpy(&buf[myString.length()], str.c_str(), str.length());
   buf[nLen] = 0;
   myString = buf;
   delete buf;
   return *this;
}

kokkosstream& kokkosstream::operator <<(const char * st) {
   std::string str = st;
   return *this << str;
}

kokkosstream& kokkosstream::operator<<(bool __n) {
   std::string str = __n ? "1" : "0";
   return *this << str;
}
kokkosstream& kokkosstream::operator<<(short __n) {
   char buff[255];
   sprintf(buff, "%d", (int)__n);
   std::string str = buff;
   return *this << str;
}
kokkosstream& kokkosstream::operator<<(unsigned short __n) {
   char buff[255];
   sprintf(buff, "%d", (unsigned int)__n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(int __n) {
   char buff[255];
   sprintf(buff, "%d", __n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(unsigned int __n) {
   char buff[255];
   sprintf(buff, "%d", __n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(long __n) {
   char buff[255];
   sprintf(buff, "%ld", __n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(unsigned long __n) {
   char buff[255];
   sprintf(buff, "%ld", __n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(long long __n) {
   char buff[255];
   sprintf(buff, "%lld", __n);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(unsigned long long __n) {
   char buff[255];
   sprintf(buff, "%lld", __n);
   std::string str = buff;
   return *this << str;
}
kokkosstream& kokkosstream::operator<<(float __f) {
   char buff[255];
   sprintf(buff, "%f", __f);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(double __f) {
   char buff[255];
   sprintf(buff, "%f", __f);
   std::string str = buff;

   return *this << str;
}
kokkosstream& kokkosstream::operator<<(long double __f) {
   char buff[255];
   sprintf(buff, "%Lf", __f);
   std::string str = buff;

   return *this << str;
}

