#include <stdio.h>
#include "src/tester.h"


int main(int argc, char* argv[]) {
    bool cuda;

    if (strcmp(argv[1], "openmp") == 0) {
        cuda = false;
        printf("OpenMP execution.\n");
    }
    else {
        cuda = true;
        printf("CUDA execution.\n");
    }

    if (tester(cuda, argv[2], argv[3])) {
        printf("\nCongrats! All tests passed!\n");
    } else {
        printf("\nOne or more tests failed!\n");
    }

    return 0;
}
