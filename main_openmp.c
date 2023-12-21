#include <stdio.h>
#include "src/tester.h"

int main() {
    if (tester()) {
        printf("\nCongrats! All tests passed!\n");
    } else {
        printf("\nOne or more tests failed!\n");
    }

    return 0;
}
