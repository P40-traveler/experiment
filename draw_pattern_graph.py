import json
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

def draw_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20, 14))

    # 按 label_id 分配颜色（节点）
    node_color_map = {}
    for node, attr in G.nodes(data=True):
        label = attr['label']
        if label not in node_color_map:
            node_color_map[label] = len(node_color_map)
        color = f'C{node_color_map[label]}'
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=800, label=f"Node Label {label}")

    # 按 label_id 分配颜色（边）
    edge_color_map = {}
    for (u, v, attr) in G.edges(data=True):
        label = attr['label']
        if label not in edge_color_map:
            edge_color_map[label] = len(edge_color_map)
        color = f'C{edge_color_map[label]}'
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=2, arrows=True, label=f"Edge Label {label}")

    node_labels = {n: f"Tag:{attr['tag']}\nLbl:{attr['label']}" for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    edge_labels = {(u, v): f"Tag:{attr['tag']}\nLbl:{attr['label']}" for u, v, attr in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # 图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Labels")

    plt.title("Graph from p_.json")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    file_path = '/home/phy/lab/executing/pathce/patterns/glogs/p2.json'
    G = load_json_graph(file_path)
    draw_graph(G)

if __name__ == "__main__":
    main()