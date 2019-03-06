#ifndef __EMU_TEST_INTERFACE_
#define __EMU_TEST_INTERFACE_
#include <streamimport.h>
#include <string>
#include <vector>
#include <math.h>

#ifdef INCLUDE_CLASS
    #define CLASS_PUBLIC __attribute__ ((visibility ("default")))
    #define VAR_EXPR
#else
    #define CLASS_PUBLIC __attribute__ ((visibility ("hidden")))
    #define VAR_EXPR extern
#endif

namespace testing {

class CLASS_PUBLIC Test {
public:
   std::string testName();
   std::string testCategory();   
   Test();
   virtual void test_body();
protected:
   std::string m_Name;
   std::string m_Category;
};

    VAR_EXPR bool test_failed;
    VAR_EXPR std::string current_category;
    VAR_EXPR std::string current_test;
    VAR_EXPR std::vector<Test*> testList;

    Test* MakeAndRegisterTestInfo( Test* testImpl); 
    void run_test_cases();
    void initialize_tests();

}

#define GET_TEST_STRING(T) #T
#define GET_CLASS_NAME(T,P) T##_##P##_Test

#define TEST_F(category,unit_test) \
   class GET_CLASS_NAME(category,unit_test) : public ::testing::Test { \
   public:\
   GET_CLASS_NAME(category,unit_test)();\
   static testing::Test* const test_info_;\
   virtual void test_body();\
};\
inline \
GET_CLASS_NAME(category,unit_test)::GET_CLASS_NAME(category,unit_test)() : ::testing::Test() { \
   m_Name = (std::string)GET_TEST_STRING(unit_test);\
   m_Category = (std::string)GET_TEST_STRING(category); \
}\
testing::Test* const GET_CLASS_NAME(category,unit_test)\
  ::test_info_ =\
    testing::MakeAndRegisterTestInfo(new GET_CLASS_NAME(category,unit_test)());\
\
inline \
void GET_CLASS_NAME(category,unit_test)::test_body()

#define ASSERT_EQ( A, B) \
{  \
   if ( A != B ) { \
      printf("[%s] ASSERT_EQ \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_NE( A, B) \
{  \
   if ( A == B ) { \
      printf("[%s] ASSERT_NE \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_LT( A, B) \
{  \
   if ( A >= B ) { \
      printf("[%s] ASSERT_LT \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_LE( A, B) \
{  \
   if ( A > B ) { \
      printf("[%s] ASSERT_LE \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_GE( A, B) \
{  \
   if ( A < B ) { \
      printf("[%s] ASSERT_GE \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}


#define ASSERT_GT( A, B) \
{  \
   if ( A <= B ) { \
      printf("[%s] ASSERT_EQ \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_FLOAT_EQ( A, B) \
{  \
   if ( fabs( (double)(A) - (double)(B) ) > 0.00000001 ) { \
      printf("[%s] ASSERT_FLOAT_EQ \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_DOUBLE_EQ( A, B) \
{  \
   if ( fabs( (double)(A) - (double)(B) ) > 0.00000001 ) { \
      printf("[%s] ASSERT_DOUBLE_EQ \n", testing::current_test.c_str() ); \
      testing::test_failed = true;  \
   } \
}

#define ASSERT_TRUE( A ) \
   if ( !(A) ) {\
      printf("[%s] ASSERT_TRUE failed \n", testing::current_test.c_str() );\
      testing::test_failed = true; \
   }

#define ASSERT_FALSE( A ) \
   if ( (A) ) {\
      printf("[%s] ASSERT_FALSE failed \n", testing::current_test.c_str() );\
      testing::test_failed = true; \
   }

#define EXPECT_EQ( A,B ) \
   if ( (A != B ) ) {\
      printf("[%s] EXPECT_EQ failed \n", testing::current_test.c_str() );\
      testing::test_failed = true; \
   }


#endif
