//
// Created by calin on 13-Dec-23.
//


#include "tester.h"


bool tester(bool cuda) {
    // Prepare directory and file variables
    char dir_in[] = "../data/infoarena_processed";
    char dir_ok[] = "../data/infoarena";

    DIR *dfd_in;
    struct dirent *dp;
    char file_path_in[100];

    bool all_tests = true;
    bool current_test = true;

    dfd_in = opendir(dir_in);

    // Iterate through directory
    while ((dp = readdir(dfd_in)) != NULL) {
        struct stat stbuf;
        sprintf(file_path_in, "%s/%s", dir_in, dp->d_name);

        // Skip files not ending in ".in"
        if (EndsWith(file_path_in, ".in") == 0) continue;

        // Skip files that can't be opened
        if (stat(file_path_in, &stbuf) == -1) continue;

        // Read graph and perform bellman_ford
        graph *g = graph_file_reader(file_path_in);
        bool result;

        if (cuda)
            cu_bellman_ford(g, &g->nodes[0]);
        else
            bellman_ford(g, &g->nodes[0]);

        // Read ok file
        char file_path_ok[100];
        char *file_name = strrchr(file_path_in, '/');
        file_name[strlen(file_name) - 3] = '\0';
        sprintf(file_path_ok, "%s%s.ok", dir_ok, file_name);

        char *ok_file_content = file_reader(file_path_ok);

        if (strcmp("Ciclu negativ!\n", ok_file_content) == 0) {
            // Negative cycle situation
            if (!result) {
                printf("%s passed!\n", file_name);
            } else {
                all_tests = false;
                printf("%s failed!\n", file_name);
            }
        } else {
            // Good graph situation
            int current_char = 0;
            for (int i = 1; i < g->n; i++) {
                int val = 0;

                // Solve the negative number situation
                bool negative = false;

                if (ok_file_content[current_char] == '-') {
                    negative = true;
                    current_char++;
                }

                // Read value from file
                while (('0' <= ok_file_content[current_char]) && (ok_file_content[current_char] <= '9')) {
                    val *= 10;
                    val += (ok_file_content[current_char] - '0');
                    current_char++;
                }
                current_char++;
                if (negative) val = -val;

                // Check values from file and from resulted graph
                if (val != g->nodes[i].d) {
                    all_tests = false;
                    current_test = false;
                    printf("%s failed!\n", file_name);
                    break;
                }
            }

            if (current_test) printf("%s passed!\n", file_name);
        }
    }

    return all_tests;
}
