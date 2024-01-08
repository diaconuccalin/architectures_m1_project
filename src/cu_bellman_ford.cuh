//
// Created by calin on 22-Dec-23.
//

#ifndef ARCHITECTURES_M1_PROJECT_CU_BELLMAN_FORD_CUH
#define ARCHITECTURES_M1_PROJECT_CU_BELLMAN_FORD_CUH

#include <stdlib.h>

#include "graph_structs.h"
#include "timing.h"
#include "utils.h"

bool cu_bellman_ford(graph *G, node *s);

#endif //ARCHITECTURES_M1_PROJECT_CU_BELLMAN_FORD_CUH
