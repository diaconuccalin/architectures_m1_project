//
// Created by diaco on 07/12/2023.
//

#include <string.h>
#include "utils.h"

node *find_node_by_name(char *node_name, node *node_list, int list_len) {
    node *current_node = node_list;
    for(int i = 0; i < list_len; i ++) {
        if (strcmp(current_node->name, node_name) == 0) {
            return current_node;
        }
        current_node += sizeof(node);
    }

    return NULL;
}
