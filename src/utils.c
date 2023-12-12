//
// Created by diaco on 07/12/2023.
//

#include <string.h>
#include "utils.h"

node *find_node_by_name(char *node_name, node *node_list, int list_len) {
    for(int i = 0; i < list_len; i ++) {
        if (strcmp(node_list[i].name, node_name) == 0) {
            return &node_list[i];
        }
    }

    return NULL;
}
