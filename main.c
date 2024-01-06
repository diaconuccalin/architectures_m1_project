#include <omp.h>
#include <stdio.h>
#include "src/tester.h"


int main() {
    bool cuda = true;
  
    // TODO: Evaluate multiple values
  
    if (tester(cuda)) {
        printf("\nCongrats! All tests passed!\n");
    } else {
        printf("\nOne or more tests failed!\n");
    }

    return 0;
}
