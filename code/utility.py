import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_edges(coef_list):
    '''Generates a dataframe of weights/connections from a NN's weight matrix.
    Note: This function also includes information on parent/child layer/neurons
    
    Args:
        coef_list (array list): List of each layer's weight matrix
    
    Returns:
        weights (DataFrame): DataFrame containing the weight associated to 
            each of the network's connection (including origin/destination data)
    '''
    data_list = []
    for layer_idx, layer in enumerate(coef_list):
        for parent_neuron in range(layer.shape[0]):
            for child_idx, weight in enumerate(layer[parent_neuron,:]):
                data_list.append([
                    layer_idx,        # parent layer
                    parent_neuron,    # parent neuron
                    layer_idx + 1,    # child layer
                    child_idx,        # child neuron
                    weight            # weight                
                ])
    weights = pd.DataFrame(data_list, columns = ['parent_layer', 
                                                 'parent_neuron', 
                                                 'child_layer', 
                                                 'child_neuron', 
                                                 'weight'])
    return weights

def get_weights(coef_list):
    '''Generates a dataframe of weights/connections from a NN's weight matrix.
    This weights array is only used to compute the weight delta between iterations.
    
    Args:
        coef_list (array list): List of each layer's weight matrix
    
    Returns:
        weights (DataFrame): DataFrame containing the weight associated to 
            each of the network's connection    
    '''
    data_list = []
    for layer_idx, layer in enumerate(coef_list):
        for parent_neuron in range(layer.shape[0]):
            for child_idx, weight in enumerate(layer[parent_neuron,:]):
                data_list.append(weight)          # weight                
    weights = np.array(data_list)
    return weights

def get_coordinate_dict(coordinates):
    '''Returns a dictionary from a DataFrame of neurons and (x,y) positions.
    It is used for quicker look up operations
    
    Args:
        coordinates (DataFrame): DataFrame containing neurons and (x,y) positions
        
    Returns
        coordinate_dict (dict): Dictionary containing (layer,neuron):{'x':pos,'y':pos} pairs
    '''
    coordinate_dict = {}
    for idx, row in coordinates.iterrows():
        coordinate_dict[(row['layer'], row['neuron'])] = {'x':row['x'], 'y':row['y']}
    return coordinate_dict


def get_neurons(weights, height=100, width=100, neuron_diameter=2, plot=False):
    '''Returns a DataFrame and Dictionary containing neuron (x,y) positions
    
    Args:
        weights (array list): List of the NN's weight matrices
        height (int): height of the chart
        width (int): width of the chart
        neuron_diameter (int): neuron diameter on the chart
        plot (bool): flag used to quickly plot the NN for debugging purposes
    
    '''
    # List of layers
    layer_list = []
    # Iterating over the weight matrices to get the layer list
    for idx, wieght_matrix in enumerate(weights):
        if idx == 0:
            layer_list.append(wieght_matrix.shape[0])
        layer_list.append(wieght_matrix.shape[1])
    # List storing future dataframe values
    data_list = []
    neuron_id = 0
    for layer_idx, layer in enumerate(layer_list):
        layer_x = (layer_idx+1)*1/(len(layer_list)+1)
        layer_x *= width
        # Computing padding based on the number of neurons per layer
        layer_padding = (height-layer*neuron_diameter)/(layer+1)
        for neuron in range(layer):
            neuron_y = height - layer_padding*(neuron+1) - neuron_diameter/2 - neuron*neuron_diameter
            # Adding data to the list, the x,y positions of each neurons
            data_list.append([neuron_id,layer_idx,neuron,layer_x, neuron_y])
            neuron_id += 1
    # Creating a dataframe
    neurons_df = pd.DataFrame(data_list, columns = ['id', 'layer', 'neuron', 'x', 'y'])
    if plot:
        plt.scatter(neurons_df['x'], neurons_df['y'])
    # Generating a dictionary from this dataframe using the get_coordinate_dict function
    neurons_dict = get_coordinate_dict(neurons_df)
    return neurons_df, neurons_dict
