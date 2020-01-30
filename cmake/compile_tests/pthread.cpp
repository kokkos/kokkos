#include <pthread.h>

void* kokkos_test(void* args) { return args; }

int main(void) {
  pthread_t thread;
  pthread_create(&thread, nullptr, kokkos_test, nullptr);
  pthread_join(thread, nullptr);
  return 0;
}
