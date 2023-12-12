//
// Created by diaco on 20/11/2023.
//

#ifndef ARCHITECTURES_M1_PROJECT_GRAPH_STRUCTS_H
#define ARCHITECTURES_M1_PROJECT_GRAPH_STRUCTS_H

typedef struct{
    char *name;
    int d;
    char *pi;
} node;

typedef struct{
    node *source;
    node *destination;
    int weight;
} edge;

typedef struct {
    // n - number of nodes
    // m - number of edges
    int n, m;

    // nodes - pointer to a list of n nodes
    node* nodes;

    // edges - pointer to a list of m edges
    edge* edges;
} graph;

#endif //ARCHITECTURES_M1_PROJECT_GRAPH_STRUCTS_H
