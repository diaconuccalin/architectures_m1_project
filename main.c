#include <stdio.h>
#include "src/graph_structs.h"
#include "src/graph_file_reader.h"

int main() {
    graph *result = graph_file_reader("..\\data\\data_0.in");

    for (int i = 0; i < result->m; i++) {
        printf("%s %s %d\n", result->edges[i].source->name, result->edges[i].destination->name, result->edges[i].weight);
    }

    return 0;
}
