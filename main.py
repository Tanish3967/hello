import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from io import BytesIO
import base64

# Node class as in the previous code
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Calculate the Manhattan distance heuristic."""
    return abs(a - b)


def a_star_search(adj_matrix, start, goal, heuristic_func):
    """Performs A* search."""
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for neighbor in range(len(adj_matrix)):
            if adj_matrix[current_node.position][neighbor] == 1 and neighbor not in closed_set:
                neighbor_node = Node(neighbor, current_node)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = heuristic_func(neighbor, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                if any(open_node for open_node in open_list if neighbor_node == open_node and neighbor_node.g >= open_node.g):
                    continue

                heapq.heappush(open_list, neighbor_node)

    return None


def uniform_cost_search(adj_matrix, start, goal):
    """Performs Uniform Cost Search."""
    open_list = []
    closed_set = set()
    start_node = Node(start)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for neighbor in range(len(adj_matrix)):
            if adj_matrix[current_node.position][neighbor] == 1 and neighbor not in closed_set:
                neighbor_node = Node(neighbor, current_node)
                neighbor_node.g = current_node.g + 1
                neighbor_node.f = neighbor_node.g

                if any(open_node for open_node in open_list if neighbor_node == open_node and neighbor_node.g >= open_node.g):
                    continue

                heapq.heappush(open_list, neighbor_node)

    return None


def generate_connected_graph(num_nodes, edge_weight):
    """Generate a connected graph with a random adjacency matrix."""
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        adj_matrix[i][i + 1] = edge_weight
        adj_matrix[i + 1][i] = edge_weight

    for i in range(num_nodes):
        for j in range(i + 2, num_nodes):
            if random.random() < 0.3:
                adj_matrix[i][j] = edge_weight
                adj_matrix[j][i] = edge_weight

    return adj_matrix


def create_networkx_graph(adj_matrix):
    """Create a NetworkX graph from an adjacency matrix."""
    G = nx.Graph()
    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])
    return G


def plot_graph(G, path=None):
    """Generate a plotly plot for the graph."""
    pos = nx.spring_layout(G)
    edges = G.edges()
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right')
        ))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        title='Graph Visualization',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig


# Streamlit Interface
def main():
    st.title("Graph Search Algorithm Dashboard")

    # Inputs for graph parameters
    num_nodes = st.slider("Number of Nodes", min_value=5, max_value=20, value=10)
    edge_weight = st.slider("Edge Weight", min_value=1, max_value=5, value=1)
    
    # Select Algorithm
    algorithm = st.selectbox(
        "Select Algorithm",
        options=["A* Search", "Uniform Cost Search", "Both"]
    )
    
    # Select Start and Goal Nodes
    start_node = st.selectbox("Start Node", options=range(num_nodes))
    goal_node = st.selectbox("Goal Node", options=range(num_nodes))

    # Generate the graph
    adj_matrix = generate_connected_graph(num_nodes, edge_weight)
    G = create_networkx_graph(adj_matrix)

    st.write(f"Adjacency Matrix for {num_nodes} nodes:")
    st.write(adj_matrix)

    # Perform Search Based on Selected Algorithm
    if algorithm == "A* Search" or algorithm == "Both":
        a_star_path = a_star_search(adj_matrix, start_node, goal_node, heuristic)
        if a_star_path:
            st.write(f"A* Path found: {a_star_path}")
        else:
            st.write("No path found using A*")

    if algorithm == "Uniform Cost Search" or algorithm == "Both":
        ucs_path = uniform_cost_search(adj_matrix, start_node, goal_node)
        if ucs_path:
            st.write(f"UCS Path found: {ucs_path}")
        else:
            st.write("No path found using UCS")

    # Plot Graph and Path
    fig = plot_graph(G, path=a_star_path if algorithm == "A* Search" or algorithm == "Both" else ucs_path)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
