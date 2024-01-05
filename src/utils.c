//
// Created by diaco on 07/12/2023.
//

#include "utils.h"


node *find_node_by_name(char *node_name, node *node_list, int list_len) {
    for (int i = 0; i < list_len; i++) {
        if (strcmp(node_list[i].name, node_name) == 0) {
            return &node_list[i];
        }
    }

    return NULL;
}


int find_node_id_by_name(char *node_name, node *node_list, int list_len) {
    for (int i = 0; i < list_len; i++) {
        if (strcmp(node_list[i].name, node_name) == 0) {
            return i;
        }
    }

    return -1;
}


int EndsWith(const char *str, const char *suffix) {
    if (!str || !suffix)
        return 0;

    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);

    if (lensuffix > lenstr)
        return 0;

    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}


// Simple file reader that returns a char containing the entire text in the file
char *file_reader(char *fileName) {
    FILE *file = fopen(fileName, "r");
    char *code;
    size_t n = 0;
    int c;

    if (file == NULL)
        return NULL;

    fseek(file, 0, SEEK_END);
    long f_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    code = malloc(f_size);

    while ((c = fgetc(file)) != EOF) {
        code[n++] = (char) c;
    }

    code[n] = '\0';

    return code;
}
