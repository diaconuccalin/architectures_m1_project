#include <omp.h>
#include <stdio.h>
#include "src/tester.h"


int main() {
    // TODO: Evaluate multiple values
    omp_set_num_threads(4);

    if (tester()) {
        printf("\nCongrats! All tests passed!\n");
    } else {
        printf("\nOne or more tests failed!\n");
    }

    return 0;
}
