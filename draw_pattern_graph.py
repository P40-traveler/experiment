import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def load_json_graph(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()

    for idx, vertex in enumerate(data.get('vertices', [])):
        tag_id = vertex['tag_id']
        label_id = vertex['label_id']
        G.add_node(idx, tag=tag_id, label=label_id)

    for edge in data.get('edges', []):
        src = edge['src']
        dst = edge['dst']
        tag_id = edge['tag_id']
        label_id = edge['label_id']
        G.add_edge(src, dst, tag=tag_id, label=label_id)

    return G

def draw_graph(G, output_path):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20, 14))

    node_color_map = {}
    for node, attr in G.nodes(data=True):
        label = attr['label']
        if label not in node_color_map:
            node_color_map[label] = len(node_color_map)
        color = f'C{node_color_map[label]}'
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=800, label=f"Node Label {label}")

    edge_color_map = {}
    for (u, v, attr) in G.edges(data=True):
        label = attr['label']
        if label not in edge_color_map:
            edge_color_map[label] = len(edge_color_map)
        color = f'C{edge_color_map[label]}'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=2, arrows=True, label=f"Edge Label {label}")

    node_labels = {n: f"Tag:{attr['tag']}\nLbl:{attr['label']}" for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=30)

    edge_labels = {(u, v): f"Tag:{attr['tag']}\nLbl:{attr['label']}" for u, v, attr in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=30)

    # 图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Labels")

    plt.title("Graph from p_.json")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path) 
    plt.close()

def main():
    directory = '/home/phy/lab/executing/pathce/patterns/glogs/'
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {directory}.")
        return

    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            G = load_json_graph(file_path)
            output_file = os.path.splitext(file_path)[0] + '.png'
            draw_graph(G, output_file)
            print(f"Graph saved to {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()