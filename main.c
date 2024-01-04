#include <stdio.h>
#include "src/tester.h"


int main() {
    bool cuda = true;
    if (tester(cuda)) {
        printf("\nCongrats! All tests passed!\n");
    } else {
        printf("\nOne or more tests failed!\n");
    }

    return 0;
}
