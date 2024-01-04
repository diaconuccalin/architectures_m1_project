//
// Created by calin on 04-Jan-24.
//

#include "timing.h"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double) ts.tv_nsec / 1e9);
}
