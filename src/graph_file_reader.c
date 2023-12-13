//
// Created by diaco on 06/12/2023.
//


#include "graph_file_reader.h"


char *read_node_name(const char *p) {
    char *node_name = malloc(5 * sizeof(char));

    for (int j = 0; j < 4; j++) {
        node_name[j] = p[j];
    }
    node_name[4] = '\0';

    return node_name;
}


graph *text_to_graph(char *text) {
    // Prepare resulting graph
    graph *to_return = malloc(sizeof(graph));

    // Prepare ints to store the number of nodes and edges
    int n = -1;
    int m = -1;

    // Read the first 2 int values and store them in n and m
    char *p = text;
    while (*p) {
        if (isdigit(*p)) {
            long val = strtol(p, &p, 10);

            if (n == -1) {
                n = val;
            } else {
                m = val;
                p++;
                break;
            }
        } else {
            p++;
        }
    }

    node *nodes = malloc(n * sizeof(node));
    for (int i = 0; i < n; i++) {
        nodes[i].name = read_node_name(p);
        p += 5;
    }

    edge *edges = malloc(m * sizeof(edge));
    for (int i = 0; i < m; i++) {
        // Read source name
        char *source_name = read_node_name(p);
        node *source = find_node_by_name(source_name, nodes, n);
        free(source_name);
        p += 5;

        // Read destination name
        char *dest_name = read_node_name(p);
        node *destination = find_node_by_name(dest_name, nodes, n);
        free(dest_name);
        p += 5;

        // Read weight
        int weight;
        while (*p) {
            if (isdigit(*p) || ((*p == '-' || *p == '+') && isdigit(*(p + 1)))) {
                weight = strtol(p, &p, 10);
            } else {
                p++;
                break;
            }
        }

        // Store
        edges[i].source = source;
        edges[i].destination = destination;
        edges[i].weight = weight;
    }

    to_return->n = n;
    to_return->m = m;
    to_return->nodes = nodes;
    to_return->edges = edges;

    return to_return;
}


graph *graph_file_reader(char *fileName) {
    // Read file content
    char *text = file_reader(fileName);

    // Generate graph
    graph *to_return = text_to_graph(text);

    // Free memory used to store file content
    free(text);

    return to_return;
}
