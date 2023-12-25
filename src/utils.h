//
// Created by diaco on 07/12/2023.
//

#ifndef ARCHITECTURES_M1_PROJECT_UTILS_H
#define ARCHITECTURES_M1_PROJECT_UTILS_H

#include <string.h>
#include <stdio.h>
#include <malloc.h>
#include "graph_structs.h"

node *find_node_by_name(char *node_name, node *node_list, int list_len);
int find_node_id_by_name(char *node_name, node *node_list, int list_len);
int EndsWith(const char *str, const char *suffix);
char *file_reader(char *fileName);

#endif //ARCHITECTURES_M1_PROJECT_UTILS_H
