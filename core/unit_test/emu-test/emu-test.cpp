#define INCLUDE_CLASS
#include <emu-test.h>
#include <memoryweb/intrinsics.h>

#include <time.h>


testing::Test::Test() {
}

std::string testing::Test::testName() {
   return m_Name;
}
std::string testing::Test::testCategory() {
   return m_Category;
}

void testing::Test::test_body() {
}

void testing::initialize_tests() {
   testing::current_test = "";
   testing::current_category = "";
   testing::test_failed = false;
}

void testing::run_test_cases() {

   FENCE();
   for (int i = 0; i < testing::testList.size(); i++) {
       testing::Test* pTest = (Test*)testing::testList[i];

       testing::current_test = pTest->testName();
       testing::current_category = pTest->testCategory();

       testing::test_failed = false;

       printf("[   RUN    ]%s::%s \n", testing::current_test.c_str(), testing::current_category.c_str());

       fflush(stdout);
       long nStart = clock();

       pTest->test_body();

       if (testing::test_failed) {
          printf("[  FAILED  ]%s::%s \n", testing::current_test.c_str(), testing::current_category.c_str());
      } else {
          printf("[  OK      ]%s::%s \n", testing::current_test.c_str(), testing::current_category.c_str());
      }
      long nEnd = clock();
      printf("[          ] %lf seconds \n", ((double)(nEnd-nStart)/(double)1000000));
      FENCE();
   }
}

testing::Test* testing::MakeAndRegisterTestInfo( testing::Test* testImpl) {
    printf("registering test: %s \n", testImpl->testName().c_str());
    testing::testList.push_back(testImpl);
    return testImpl;
}

