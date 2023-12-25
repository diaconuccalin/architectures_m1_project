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
        edge *e = &G->edges[index];

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

    graph *d_G;
    node *d_nodes;
    edge *d_edges;

    // Graph
    cudaMalloc((void **) &d_G, sizeof(graph));
    cudaMemcpy(d_G, G, sizeof(graph), cudaMemcpyHostToDevice);

    // Nodes
    cudaMalloc((void**)&d_nodes, G->n * sizeof(node));
    for (int i = 0; i < G->n; i++) {
        // Name
        char *d_node_name;
        cudaMalloc((void **) &d_node_name, 5 * sizeof(char));
        cudaMemcpy(d_node_name, G->nodes[i].name, 5 * sizeof(char), cudaMemcpyHostToDevice);

        // Node
        cudaMemcpy(&d_nodes[i], &G->nodes[i], sizeof(node), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_nodes[i].name, &d_node_name, sizeof(char*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&d_G->nodes, &d_nodes, sizeof(node*), cudaMemcpyHostToDevice);

    // Edges
    cudaMalloc((void**)&d_edges, G->m * sizeof (edge));
    for (int i = 0; i < G->m; i++) {
        int source_index = find_node_id_by_name(G->edges[i].source->name, G->nodes, G->m);
        int destination_index = find_node_id_by_name(G->edges[i].destination->name, G->nodes, G->m);

        // Source
        node *d_source_node;
        cudaMalloc((void **) &d_source_node, sizeof(node));
        cudaMemcpy(d_source_node, &(d_nodes[source_index]), sizeof(node), cudaMemcpyHostToDevice);

        // Destination
        node *d_destination_node;
        cudaMalloc((void **) &d_destination_node, sizeof(node));
        cudaMemcpy(d_destination_node, &(d_nodes[destination_index]), sizeof(node), cudaMemcpyHostToDevice);

        // Edge
        cudaMemcpy(&d_edges[i], &G->edges[i], sizeof(edge), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_edges[i].source, &d_source_node, sizeof(node*), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_edges[i].destination, &d_destination_node, sizeof(node*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&d_G->edges, &d_edges, sizeof(edge*), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    end = gettime2();
    printf("Alloc & copy time: %f\n", end - start);

    int grid_dim = int(sqrt(G->m));
    int block_dim = (G->m / grid_dim + 1);

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
