//
// Created by calin on 12-Dec-23.
// Implementation from Introduction to Algorithms, Cormen et al.
//


#include "bellman_ford.h"


void initialize_single_source(graph *G, node *s) {
    for (int i = 0; i < G->n; i++) {
        G->nodes[i].d = INT_MAX;
        G->nodes[i].pi = NULL;
    }

    s->d = 0;
}


void relax(edge *e) {
    node *u = e->source;
    node *v = e->destination;
    int new_d = u->d + e->weight;

    if (v->d > new_d) {
        v->d = new_d;
        v->pi = u->name;
    }
}


bool bellman_ford(graph *G, node *s) {
    initialize_single_source(G, s);

    double start = get_time();
    for (int i = 1; i < G->n; i++) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int j = 0; j < G->m; j++) {
            if (G->edges[j].source->d < INT_MAX)
                relax(&G->edges[j]);
        }
    }
    double end = get_time();
    printf("Parallel section: %.6f seconds\n", end - start);

    for (int i = 0; i < G->m; i++) {
        edge *current_edge = &G->edges[i];
        node *u = current_edge->source;
        node *v = current_edge->destination;

        if (v->d > (u->d + current_edge->weight))
            return false;
    }

    return true;
}
