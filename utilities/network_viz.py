import matplotlib.pyplot as plt
from utilities.Convenience import unrestrained_residues 
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import pandas as pd


def create_weighted_network(matrix,outfile="networkimage",color_components=False,electrostatics=False):
    """ plots and returns an igraph network object 

    Parameters
    ----------
    matrix:np.ndarray, shape=(n_residues,n_residues)
        An adjacency matrix created to represent all interactions in a system. Size should be representative
        of pairwise comparisons between every residue in the system with the weights representing some measure of interaction,
        whether it be hydrogen bonding, electrostatic interactions or even distance. The assumption however is that this is a 
        weighted undirected adjacency matrix.
    
    Returns
    -------
    graph:igraph.Graph,


    Notes
    ------
    We do assume there are indexes included in theese matrices so I, implicitly ignore them at the start of the script
        this line-(matrix=matrix[1:,1:])
    
    


    Examples
    --------



    """
    #Create matrix from adjacency and re-assign label names
    node_labels=matrix[0,1:]
    matrix=matrix[1:,1:]

    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    labels_dict = {i: str(int(node_labels[i])) for i in range(len(node_labels))}
    G = nx.relabel_nodes(G, labels_dict)

    # Remove edges with zero weight
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)

    #incase of negative vals and then assigning label names
    if electrostatics == False:
        # Customize edge colors based on weights
        edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in G.edges()]

    if color_components:
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Create a colormap (use a colormap with distinct colors)
        colormap = plt.cm.tab20
        
        # If there are many components, you might need more colors
        if len(components) > 20:
            colors = [mcolors.to_hex(random.choice(list(mcolors.CSS4_COLORS.values()))) 
                      for _ in range(len(components))]
        else:
            colors = [mcolors.to_hex(colormap(i)) for i in range(len(components))]
        
        # Create a dictionary mapping each node to its color based on component
        color_map = {}
        for component_id, component in enumerate(components):
            for node in component:
                color_map[node] = colors[component_id % len(colors)]
        
        # Get node colors in the order of G.nodes()
        node_colors = [color_map[node] for node in G.nodes()]
        
        # Draw the graph with nodes colored by component
        nx.draw(G, pos=pos,
                node_color=node_colors,
                with_labels=True, 
                node_size=200, 
                font_size=10, 
                font_color="black", 
                edge_color=edge_colors, 
                width=2)
        
        # Add a legend for component colors
        for i, color in enumerate(colors[:len(components)]):
            plt.plot([0], [0], 'o', color=color, 
                    label=f'Component {i+1} ({len(components[i])} nodes)')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        plt.title(f"Network with {len(components)} Connected Components")

    # Plot the graph with customized visual style
    plt.figure(figsize=(13, 13))

    # Draw the graph
    nx.draw(G, with_labels=True, node_size=60, font_size=6, font_color="black", edge_color=edge_colors, width=1)

    # Save and show the plot
    plt.savefig(outfile)
    plt.show()

    return G

def draw_colored_components(G, outfile="colored_network.png"):
    # Find connected components
    components = list(nx.connected_components(G))
    
    # Create a colormap 
    colormap = plt.cm.tab20  # Use a colormap with distinct colors
    
    # If there are many components, you might need more colors
    if len(components) > 20:
        # Create a list of random colors
        colors = [mcolors.to_hex(random.choice(list(mcolors.CSS4_COLORS.values()))) 
                  for _ in range(len(components))]
    else:
        # Use colors from the colormap
        colors = [mcolors.to_hex(colormap(i)) for i in range(len(components))]
    
    # Create a dictionary mapping each node to its color based on component
    color_map = {}
    for component_id, component in enumerate(components):
        for node in component:
            color_map[node] = colors[component_id % len(colors)]
    
    # Get node colors in the order of G.nodes()
    node_colors = [color_map[node] for node in G.nodes()]
    
    # Calculate positions
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
    
    # Plot the graph
    plt.figure(figsize=(12, 12))
    
    # Draw nodes colored by component
    nx.draw(G, pos, 
            node_color=node_colors,
            with_labels=True, 
            node_size=200, 
            font_size=10, 
            font_color="black", 
            width=2)
    
    # Add a legend for component colors
    for i, color in enumerate(colors[:len(components)]):
        plt.plot([0], [0], 'o', color=color, label=f'Component {i+1} ({len(components[i])} nodes)')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.title(f"Network with {len(components)} Connected Components")
    
    # Save and show the plot
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    
    return components

def detect_components(graph,title="Network Components",outfile="Networkimage"):
    """ Returns none
    
    plots and returns an igraph network object 

    Parameters
    ----------
    graph:igraph.Graph,
        An Igraph object upon which to preform the desired component detection and visualization 

    title:str,default=Network Components"
        The title you would like to be displayed at the top of the produced graph

    outfile:str,default=Networkimage"
        The outfile name for saving the detected components of the graph
    
    Returns
    -------
    graph:igraph.Graph,


    Examples
    --------


    """

    # Assuming corellation_graph and CCU_GCU_avgtraj are defined

    components = graph.connected_components()

    # Assign a unique color to each component using a colormap
    num_components = len(components)
    cmap = plt.cm.rainbow  # You can choose a different colormap

    # Create a dictionary to map each node to a color based on its component
    node_colors = []
    for node in range(len(graph.vs)):
        component_idx = next(i for i, component in enumerate(components) if node in component)
        # Use the colormap to get a color for the node based on its component
        node_colors.append(cmap(component_idx / num_components))  # Normalize the component index

    # Customize the plot by changing edge width and color based on weights, and add node labels
    visual_style = {
        "edge_color": ["red" if graph.es[i]["weight"] < 0 else "blue" for i in range(len(graph.es))],  # Red for negative, blue for positive
        "vertex_size": 20,
        "vertex_label_size": 10,
        "vertex_color": node_colors,  # Color nodes by component
    }

    # Plot the graph with the customized style
    # Plot the graph with customized visual style
    fig, ax = plt.subplots(figsize=(13, 13))
    plot(graph, target=ax, **visual_style)
    plt.title(title) 
    plt.savefig(outfile)
    plt.show()

def extract_measures(G):
    """ Returns none
    
    plots and returns an igraph network object 

    Parameters
    ----------
    graph:igraph.Graph,
        An Igraph object upon which to preform the desired component detection and visualization 

    title:str,default=Network Components"
        The title you would like to be displayed at the top of the produced graph

    outfile:str,default=Networkimage"
        The outfile name for saving the detected components of the graph
    
    Returns
    -------
    graph:igraph.Graph,


    Examples
    --------


    """
    metrics = {
        "degree": dict(G.degree()),
        "betweenness": nx.betweenness_centrality(G),#shortest path between nodes
        "closeness": nx.closeness_centrality(G), #average length of shortest path from node to all other nodes in the network
        "degree_centrality": nx.degree_centrality(G),#fraction of nodes this node is connected to
        "clustering": nx.clustering(G), #how connected the neighbores of a node are to each other
    }

    metricsdf = pd.DataFrame(metrics)

    return metricsdf

def find_matching_residues(df, residues_in_important_pairing, top_n=10):
    """
    Finds residues that match between the top N indexes of each column in the DataFrame
    and a predefined list of residues.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.
    residues_in_important_pairing : list
        A list of residues to compare against.
    top_n : int, optional
        The number of top indexes to extract from each column (default is 10).

    Returns
    -------
    dict
        A dictionary where keys are column names and values are lists of matching residues.
    """
    # Initialize an empty dictionary to store the results
    matching_residues_dict = {}

    # Iterate through columns of the DataFrame
    for column in df.columns:
        # Sort the DataFrame by the current column and get the top N indexes
        top_n_indexes = df.sort_values(by=column, ascending=False).head(top_n).index.tolist()
        
        # Convert indexes to residues (assuming indexes correspond to residues)
        top_n_residues = [str(index) for index in top_n_indexes]
        
        # Find matching residues between top_n_residues and residues_in_important_pairing
        matching_residues = [residue for residue in top_n_residues if residue in residues_in_important_pairing]
        
        # Add the matching residues to the dictionary with the column name as the key
        if matching_residues:  # Only add if there are matching residues
            matching_residues_dict[column] = matching_residues

    return matching_residues_dict

def highlight_edges(G, edges_to_highlight, highlight_color="gold", default_color="black", outfile=None):
    """
    Highlights specified edges in an undirected NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
        The input undirected graph.
    edges_to_highlight : list of lists
        A list of node pairs to highlight.
    highlight_color : str, optional
        Color for the highlighted edges (default is 'gold').
    default_color : str, optional
        Color for all other edges (default is 'black').
    outfile : str, optional
        Path to save the graph image (default is None).

    Returns
    -------
    None
    """
    
    # Create a set of frozenset edges to handle undirected graph edge ordering
    highlight_set = {frozenset(tuple(pair)) for pair in edges_to_highlight}

    # Create edge colors list
    edge_colors = [highlight_color if frozenset(edge) in highlight_set else default_color 
                   for edge in G.edges()]

    # Draw the graph
    plt.figure(figsize=(13, 13))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, 
            font_color="black", edge_color=edge_colors, width=1.5)

    # Save or display the graph
    if outfile:
        plt.savefig(outfile, format="png", dpi=300, bbox_inches="tight")
        print(f"Graph saved to {outfile}")
    else:
        plt.show()

    plt.close()  # Close the figure to free memory

def create_weighted_highlighted_network(matrix, residue_pairs_to_color=None, outfile="networkimage", color_components=False, electrostatics=False):
    """ 
    Plots and returns a networkx network object with customizable edge coloring
    
    Parameters
    ----------
    matrix : np.ndarray, shape=(n_residues,n_residues)
        An adjacency matrix representing pairwise interactions between residues.
    residue_pairs_to_color : list of lists, optional
        A list of lists, where each sublist contains two residue labels to be colored 
        with a distinct color from other edges.
    outfile : str, optional
        Filename for saving the network visualization
    color_components : bool, optional
        Whether to color network components differently
    electrostatics : bool, optional
        Flag for handling electrostatic interactions
    
    Returns
    -------
    networkx.Graph
        The created network graph
    """
    # Create matrix from adjacency and re-assign label names
    node_labels = matrix[0,1:]
    matrix = matrix[1:,1:]
    
    # Create graph from numpy array
    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    
    # Relabel nodes with string labels
    labels_dict = {i: str(int(node_labels[i])) for i in range(len(node_labels))}
    G = nx.relabel_nodes(G, labels_dict)
    
    # Remove nodes with zero degree
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    # Default edge colors
    if electrostatics == False:
        edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in G.edges()]
    else:
        edge_colors = ['blue'] * len(G.edges())
    
    # Custom coloring for specific residue pairs
    if residue_pairs_to_color:
        # Choose a distinct color for special residue pairs
        special_edge_color = 'green'  # You can change this to any color you prefer
        
        # Modify edge colors for specified pairs
        for i, (u, v) in enumerate(G.edges()):
            for pair in residue_pairs_to_color:
                # Check if both nodes in the pair match the current edge
                if set(pair) == {u, v}:
                    edge_colors[i] = special_edge_color
    
    # Component coloring (if requested)
    if color_components:
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Create a colormap
        colormap = plt.cm.tab20
        
        # Generate colors for components
        if len(components) > 20:
            colors = [mcolors.to_hex(random.choice(list(mcolors.CSS4_COLORS.values())))
                      for _ in range(len(components))]
        else:
            colors = [mcolors.to_hex(colormap(i)) for i in range(len(components))]
        
        # Create color map for nodes
        color_map = {}
        for component_id, component in enumerate(components):
            for node in component:
                color_map[node] = colors[component_id % len(colors)]
        
        # Get node colors in order
        node_colors = [color_map[node] for node in G.nodes()]
        
        # Draw the graph with component coloring
        plt.figure(figsize=(13, 13))
        pos = nx.spring_layout(G, seed=42)  # Added layout for consistent positioning
        nx.draw(G, pos=pos,
                node_color=node_colors,
                with_labels=True,
                node_size=200,
                font_size=10,
                font_color="black",
                edge_color=edge_colors,
                width=2)
        
        # Add legend for components
        for i, color in enumerate(colors[:len(components)]):
            plt.plot([0], [0], 'o', color=color,
                     label=f'Component {i+1} ({len(components[i])} nodes)')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        plt.title(f"Network with {len(components)} Connected Components")
    else:
        # Standard graph drawing
        plt.figure(figsize=(13, 13))
        pos = nx.spring_layout(G, seed=42)  # Added layout for consistent positioning
        nx.draw(G, pos=pos, 
                with_labels=True, 
                node_size=60, 
                font_size=6, 
                font_color="black", 
                edge_color=edge_colors, 
                width=1)
    
    # Save and show the plot
    plt.savefig(outfile)
    plt.show()
    
    return G
