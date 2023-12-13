import os

from data_generator import get_random_string


def main():
    # Prepare data paths
    data_path = os.path.join("..", "data")
    infoarena_path = os.path.join(data_path, "infoarena")
    infoarena_processed_path = os.path.join(data_path, "infoarena_processed")

    # Iterate through files
    for el in os.listdir(infoarena_path):
        # Only process in files
        if el.endswith(".ok"):
            continue

        print("Processing " + el)

        # Prepare infoarena file path
        in_file_path = os.path.join(infoarena_path, el)

        with open(in_file_path, "r") as fin:
            # Read the number of nodes and edges
            first_line = fin.readline().split()
            n = int(first_line[0])
            m = int(first_line[1])

            # Randomly generate n node names
            nodes = dict()
            for i in range(1, n + 1):
                node_name = get_random_string(4)
                while node_name in nodes.values():
                    node_name = get_random_string(4)
                nodes[i] = node_name

            # Read edges
            edges = list()
            for _ in range(m):
                current_edge = fin.readline().split()
                edges.append((
                    nodes[int(current_edge[0])],
                    nodes[int(current_edge[1])],
                    current_edge[2]
                ))

            # Write to new file
            with open(os.path.join(infoarena_processed_path, el), "w") as fout:
                # Write n m
                fout.write(str(n) + " " + str(m) + "\n")

                # Write nodes names
                fout.write(' '.join(list(nodes.values())) + "\n")

                # Write edges
                for (s, d, w) in edges:
                    fout.write(' '.join([s, d, w]) + "\n")


if __name__ == "__main__":
    main()
