import os
import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def main(n, m):
    # Prepare nodes and edges list
    nodes = list()
    edges = list()

    # Randomly generate n nodes
    for _ in range(n):
        node_name = get_random_string(4)
        while node_name in nodes:
            node_name = get_random_string(4)
        nodes.append(node_name)

    # Randomly generate m edges
    for _ in range(m):
        source = random.choice(nodes)
        destination = random.choice(nodes)

        while (source == destination) or ((source, destination) in edges):
            source = random.choice(nodes)
            destination = random.choice(nodes)

        edges.append((source, destination))

    # Prepare output file
    out_path = "../data"
    file_id = 0
    while str(n) + "_" + str(m) + "_" + str(file_id) + ".in" in os.listdir(out_path):
        file_id += 1

    # Write result to file
    file_name = str(n) + "_" + str(m) + "_" + str(file_id) + ".in"
    with open(os.path.join(out_path, file_name), "w") as f:
        f.write(str(n) + " " + str(m) + "\n")

        nodes_text = ' '.join(nodes)
        f.write(nodes_text + "\n")

        for (source, destination) in edges:
            f.write(source + " " + destination + " " + str(random.randint(-10000, 10000)) + "\n")

    return None


if __name__ == "__main__":
    main(100, 1000)
