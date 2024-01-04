//
// Created by calin on 22-Dec-23.
//

#include <cstdio>

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


__global__ void cu_relax(
        int *m,
        int *nodes_ds,
        int *nodes_pis,
        int *edges_sources,
        int *edges_destinations,
        int *edges_weights) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < *m) {
        int source_index = edges_sources[index];
        int destination_index = edges_destinations[index];

        if (nodes_ds[source_index] < INT_MAX) {
            if (nodes_ds[destination_index] > (nodes_ds[source_index] + edges_weights[index])) {
                nodes_ds[destination_index] = nodes_ds[source_index] + edges_weights[index];
                nodes_pis[destination_index] = source_index;
            }
        }
    }
}


extern "C"
bool cu_bellman_ford(graph *G, node *s) {
    cu_initialize_single_source(G, s);
    double start_time = get_time();

    // Size information
    int *d_n, *d_m;

    cudaMalloc((void **) &d_n, sizeof(int));
    cudaMalloc((void **) &d_m, sizeof(int));

    cudaMemcpy(d_n, &G->n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &G->m, sizeof(int), cudaMemcpyHostToDevice);

    // Nodes information
    int *nodes_ds, *nodes_pis;
    int *d_nodes_ds, *d_nodes_pis;

    cudaMalloc((void **) &d_nodes_ds, G->n * sizeof(int));
    cudaMalloc((void **) &d_nodes_pis, G->n * sizeof(int));

    nodes_ds = (int *) malloc(G->n * sizeof(int));
    nodes_pis = (int *) malloc(G->n * sizeof(int));
    for (int i = 0; i < G->n; i++) {
        nodes_ds[i] = G->nodes[i].d;
        nodes_pis[i] = -1;
    }

    cudaMemcpy(d_nodes_ds, nodes_ds, G->n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pis, nodes_pis, G->n * sizeof(int), cudaMemcpyHostToDevice);

    // Edges information
    int *edges_sources, *edges_destinations, *edges_weights;
    int *d_edges_sources, *d_edges_destinations, *d_edges_weights;

    cudaMalloc((void **) &d_edges_sources, G->m * sizeof(int));
    cudaMalloc((void **) &d_edges_destinations, G->m * sizeof(int));
    cudaMalloc((void **) &d_edges_weights, G->m * sizeof(int));

    edges_sources = (int *) malloc(G->m * sizeof(int));
    edges_destinations = (int *) malloc(G->m * sizeof(int));
    edges_weights = (int *) malloc(G->m * sizeof(int));
    for (int i = 0; i < G->m; i++) {
        edges_sources[i] = find_node_id_by_name(G->edges[i].source->name, G->nodes, G->m);
        edges_destinations[i] = find_node_id_by_name(G->edges[i].destination->name, G->nodes, G->m);
        edges_weights[i] = G->edges[i].weight;
    }

    cudaMemcpy(d_edges_sources, edges_sources, G->m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_destinations, edges_destinations, G->m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_weights, edges_weights, G->m * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    double end_time = get_time();
    printf("Memory allocation time: %fs.\n", end_time - start_time);

    // Processing
    int grid_dim = int(sqrt(G->m));
    int block_dim = (G->m / grid_dim + 1);

    start_time = get_time();
    for (int i = 1; i < G->n; i++) {
        cu_relax<<<grid_dim, block_dim>>>(
                d_m,
                d_nodes_ds,
                d_nodes_pis,
                d_edges_sources,
                d_edges_destinations,
                d_edges_weights
        );
        cudaDeviceSynchronize();
    }
    end_time = get_time();
    printf("CUDA execution time: %fs.\n", end_time - start_time);

    cudaMemcpy(nodes_ds, d_nodes_ds, G->n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodes_pis, d_nodes_pis, G->n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < G->n; i++) {
        G->nodes[i].d = nodes_ds[i];
        G->nodes[i].pi = G->nodes[nodes_pis[i]].name;
    }

    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_nodes_ds);
    cudaFree(d_nodes_pis);
    cudaFree(d_edges_sources);
    cudaFree(d_edges_destinations);
    cudaFree(d_edges_weights);
    cudaDeviceSynchronize();

    free(nodes_ds);
    free(nodes_pis);
    free(edges_sources);
    free(edges_destinations);
    free(edges_weights);

    for (int i = 0; i < G->m; i++) {
        edge *current_edge = &G->edges[i];
        node *u = current_edge->source;
        node *v = current_edge->destination;

        if (v->d > (u->d + current_edge->weight))
            return false;
    }

    return true;
}
