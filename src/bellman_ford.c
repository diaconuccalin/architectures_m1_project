//
// Created by calin on 12-Dec-23.
// Implementation from Introduction to Algorithms, Cormen et al.
//


#include "bellman_ford.h"


void initialize_single_source(graph *G, node *s) {
    // TODO: Experiment with multiple schedule sizes (less browsing through memory if bigger chunk
    // TODO: If not doing a lot, just cancel it since it runs only once per graph
    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < G->n; i++) {
        G->nodes[i].d = INT_MAX;
        G->nodes[i].pi = NULL;
    }

    s->d = 0;
}


void relax(edge *e) {
    node *u = e->source;
    node *v = e->destination;

    if (v->d > (u->d + e->weight)) {
        v->d = u->d + e->weight;
        v->pi = u->name;
    }
}


bool bellman_ford(graph *G, node *s) {
    initialize_single_source(G, s);

    for (int i = 1; i < G->n; i++) {
        // TODO: Experiment with multiple schedule sizes and procedures
        // TODO: Try assigning chunk sizes based on graph size (up to m / N_THREADS)
        #pragma omp parallel for schedule(dynamic, 128)
        for (int j = 0; j < G->m; j++) {
            if (G->edges[j].source->d < INT_MAX)
                relax(&G->edges[j]);
        }
    }

    for (int i = 0; i < G->m; i++) {
        edge *current_edge = &G->edges[i];
        node *u = current_edge->source;
        node *v = current_edge->destination;

        if (v->d > (u->d + current_edge->weight))
            return false;
    }

    return true;
}
