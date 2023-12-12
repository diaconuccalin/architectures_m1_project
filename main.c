#include <stdio.h>
#include <stdbool.h>
#include "src/graph_structs.h"
#include "src/graph_file_reader.h"
#include "src/bellman_ford.h"

int main() {
    graph *g = graph_file_reader("..\\data\\data_0.in");
    bool result = bellman_ford(g, &g->nodes[0]);

    if (result) {
        for (int i = 0; i < g->n; i++) {
            node *current_node = &g->nodes[i];
            printf("Vertex %s: previous %s value %d\n", current_node->name, current_node->pi, current_node->d);
        }
    } else {
        printf("Reachable negative cycle found.\n");
    }

    return 0;
}
