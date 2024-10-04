import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# Initialize edge lengths
edge_lengths = {(1, 2): 2, (1, 3): 1.5, (2, 4): 1.2, (3, 4): 1.8}

# Define update function for animation
def update(frame):
    # Increase edge length for the first edge every second
    if frame % 2 == 0:
        edge_lengths[(1, 2)] += 0.1
    # Decrease edge length for the second edge every two seconds
    if frame % 4 == 0:
        edge_lengths[(2, 4)] -= 0.1
    
    # Update edge attributes
    nx.set_edge_attributes(G, edge_lengths, 'length')
    
    # Clear plot
    plt.clf()
    
    # Draw the graph with edge lengths
    pos = nx.spring_layout(G)  # Compute node positions using a spring layout algorithm
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=12, font_weight='bold', width=2, edge_color='gray')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lengths, font_color='red')
    
    plt.title("Time: {} seconds".format(frame))

# Create animation
ani = FuncAnimation(plt.gcf(), update, frames=1000, interval=10, repeat=False)

plt.show()
