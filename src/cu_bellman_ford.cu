//
// Created by calin on 22-Dec-23.
//

#include <cstdio>
#include <ctime>

extern "C" {
#include "cu_bellman_ford.cuh"
}


void cu_initialize_single_source(graph *G, node *s) {
    for (int i = 0; i < G->n; i++) {
        G->nodes[i].d = INT_MAX;
        G->nodes[i].pi = nullptr;
    }

    s->d = 0;
}


__global__ void cu_relax(graph *G) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < G->m) {
        edge *e;
        cudaMalloc((void**)&e, sizeof(edge));
        e = &G->edges[index];

        if (e->source->d < INT_MAX) {
            node *u = e->source;
            node *v = e->destination;

            if (v->d > (u->d + e->weight)) {
                v->d = u->d + e->weight;
                v->pi = u->name;
            }
        }
    }
}


double gettime2()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}


extern "C"
bool cu_bellman_ford(graph *G, node *s) {
    cu_initialize_single_source(G, s);

    graph *d_G;

    double start, end;

    start = gettime2();
    cudaMalloc((void**)&d_G, sizeof(graph));
    end = gettime2();
    printf("Alloc time: %f\n", end - start);

    start = gettime2();
    cudaMemcpy(d_G, &G, sizeof(graph), cudaMemcpyHostToDevice);
    end = gettime2();
    printf("Mem copy time: %f\n", end - start);


    int grid_dim = int(sqrt(G->m));
    int block_dim = (G->m / grid_dim + 1);
    printf("%d\n", G->m);

    start = gettime2();
    for (int i = 1; i < G->n; i++) {
        cu_relax<<<grid_dim, block_dim>>>(d_G);
    }
    end = gettime2();
    printf("Actual exe time: %f\n", end - start);

    for (int i = 0; i < G->m; i++) {
        edge *current_edge = &G->edges[i];
        node *u = current_edge->source;
        node *v = current_edge->destination;

        if (v->d > (u->d + current_edge->weight))
            return false;
    }

    cudaMemcpy(&G, d_G, sizeof(graph), cudaMemcpyDeviceToHost);
    cudaFree(d_G);
    free(G);

    return true;
}
