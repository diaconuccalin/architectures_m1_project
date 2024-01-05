//
// Created by calin on 22-Dec-23.
//

#include <cstdio>

extern "C" {
#include "cu_bellman_ford.cuh"
}


void error_handling() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
        printf("%s\n", cudaGetErrorString(error));
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
    error_handling();

    cudaMemcpy(d_n, &G->n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &G->m, sizeof(int), cudaMemcpyHostToDevice);
    error_handling();

    // Nodes information
    int *nodes_ds, *nodes_pis;
    int *d_nodes_ds, *d_nodes_pis;

    cudaMalloc((void **) &d_nodes_ds, G->n * sizeof(int));
    cudaMalloc((void **) &d_nodes_pis, G->n * sizeof(int));
    error_handling();

    cudaMallocHost((void **) &nodes_ds, G->n * sizeof(int));
    cudaMallocHost((void **) &nodes_pis, G->n * sizeof(int));
    error_handling();

    for (int i = 0; i < G->n; i++) {
        nodes_ds[i] = G->nodes[i].d;
        nodes_pis[i] = -1;
    }

    cudaMemcpy(d_nodes_ds, nodes_ds, G->n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pis, nodes_pis, G->n * sizeof(int), cudaMemcpyHostToDevice);
    error_handling();

    // Edges information
    int *edges_sources, *edges_destinations, *edges_weights;
    int *d_edges_sources, *d_edges_destinations, *d_edges_weights;

    cudaMalloc((void **) &d_edges_sources, G->m * sizeof(int));
    cudaMalloc((void **) &d_edges_destinations, G->m * sizeof(int));
    cudaMalloc((void **) &d_edges_weights, G->m * sizeof(int));
    error_handling();

    cudaMallocHost((void **) &edges_sources, G->m * sizeof(int));
    cudaMallocHost((void **) &edges_destinations, G->m * sizeof(int));
    cudaMallocHost((void **) &edges_weights, G->m * sizeof(int));
    error_handling();

    for (int i = 0; i < G->m; i++) {
        edges_sources[i] = find_node_id_by_name(G->edges[i].source->name, G->nodes, G->m);
        edges_destinations[i] = find_node_id_by_name(G->edges[i].destination->name, G->nodes, G->m);
        edges_weights[i] = G->edges[i].weight;
    }

    cudaMemcpy(d_edges_sources, edges_sources, G->m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_destinations, edges_destinations, G->m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_weights, edges_weights, G->m * sizeof(int), cudaMemcpyHostToDevice);
    error_handling();

    cudaDeviceSynchronize();

    double end_time = get_time();
    printf("\nMemory allocation time: %fs.\n", end_time - start_time);

    // Processing
    dim3 grid_dim(int(sqrt(G->m)));
    dim3 block_dim(G->m / grid_dim.x + 1);

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
        error_handling();
        cudaDeviceSynchronize();
    }
    end_time = get_time();
    printf("CUDA execution time: %fs.\n", end_time - start_time);

    cudaMemcpy(nodes_ds, d_nodes_ds, G->n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodes_pis, d_nodes_pis, G->n * sizeof(int), cudaMemcpyDeviceToHost);
    error_handling();
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
    error_handling();
    cudaDeviceSynchronize();

    cudaFreeHost(nodes_ds);
    cudaFreeHost(nodes_pis);
    cudaFreeHost(edges_sources);
    cudaFreeHost(edges_destinations);
    cudaFreeHost(edges_weights);
    error_handling();

    for (int i = 0; i < G->m; i++) {
        edge *current_edge = &G->edges[i];
        node *u = current_edge->source;
        node *v = current_edge->destination;

        if (v->d > (u->d + current_edge->weight))
            return false;
    }

    return true;
}
