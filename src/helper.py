from scipy.ndimage import binary_dilation, label
import numpy as np
import tensorflow as tf
from load_dataset import get_dataset 


class ClipConstraint(tf.keras.constraints.Constraint):#Contraint class to limit rho, r and sigma between 0 and 1
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

def split_data(data, masks, buffer_size=5, Y=None):
    """
    Splits the data into separate events based on a binary mask and buffer.

    Parameters:
     - data: arraylike structured like [events, x, y, covariates] 
     - masks: int or arraylike structured like [event, x, y], which covariate to use as a binary mask. If int, uses that column from the covariates. 
     - buffer_size: int, amount of cells to include around any 1 in the binary mask. Intersecting buffers will be combined.
     - Y: optional arraylike structured like [events, x, y]. Will be split the same way as data.
    Returns:
     - list of rectangular grid regions which include one or more events in the binary mask and their buffer
    """
    #Support for covariate based masks
    if isinstance(masks,int):
        masks = data[:,:,:,masks]

    #To store the split events
    regions = []
    Ys = [] #Only used if Y exists. 
    for event in range(data.shape[0]):
        binary_mask = masks[event]  # Extract mask for this event
        buffered_mask = binary_dilation(binary_mask, iterations=buffer_size) #create buffer
        labeled_mask, num_features = label(buffered_mask) #Connected cells get the same number starting at 1, 0 is background
        # print(labeled_mask)
        for label_idx in range(1, num_features + 1):  # Ignore background (label 0)
            indices = np.argwhere(labeled_mask == label_idx)  # Get coordinates of bounding box
            # Compute bounding 
            x_min, y_min = indices.min(axis=0)
            x_max, y_max = indices.max(axis=0)
            
            #Split and store the events
            regions.append(data[event, x_min:x_max + 1, y_min:y_max + 1, :])
            if len(Y):
                Ys.append(Y[event, x_min:x_max + 1, y_min:y_max + 1])

    return regions, Ys

def generate_grid_adjacency(data, max_distance=5, function="exponential"):
    """
    Generate an adjacency matrix for a 2D grid using a specified function and neighborhood size.
    
    Parameters:
    - data: 2D array representing the grid.
    - max_distance: Maximum distance for neighbor connections.
    - function: Defines the weighting of adjacency. Supports "linear", "exponential", "inverse_distance", or a custom function.
    
    Returns:
    - A dense adjacency matrix of shape (rows*cols, rows*cols) where rows and cols are the grid dimensions.
    """
    x_size = tf.shape(data)[0]
    y_size = tf.shape(data)[1]

    # Define default functions
    if isinstance(function, str):
        if function == "linear":
            def decay_fn(distance):
                return (-distance + tf.cast(max_distance + 1, tf.float32)) / tf.cast(max_distance, tf.float32)
        elif function == "exponential":
            def decay_fn(distance):
                return tf.exp(-distance)
        else:
            raise ValueError("Unknown function name")
    else:
        raise ValueError("Custom functions must be defined as a TensorFlow-compatible function")

    # Generate flattened grid coordinates
    rows = tf.repeat(tf.range(x_size), y_size)
    cols = tf.tile(tf.range(y_size), [x_size])

    # Stack into coordinates: shape (n_points, 2)
    coords = tf.stack([rows, cols], axis=1)

    # Compute pairwise differences
    diffs = tf.expand_dims(coords, 1) - tf.expand_dims(coords, 0)  # shape (n_points, n_points, 2)
    distances = tf.norm(tf.cast(diffs, tf.float32), axis=2)  # shape (n_points, n_points)

    # Apply decay function
    weights = decay_fn(distances)

    # Mask weights beyond max_distance
    max_d = tf.cast(max_distance, tf.float32)
    weights = tf.where(distances <= max_d, weights, tf.zeros_like(weights))

    # Zero out diagonal (no self-connections)
    weights = weights - tf.linalg.diag(tf.linalg.diag_part(weights))

    return weights


def get_dataset_generator(mode = 'train', buffer_size=5, do_spatial_split=True):
    """
    Create a dataset generator for streaming data from TFRecord files.
    
    Parameters:
        - mode: 'train', 'val', or 'test' to specify the dataset type. This is used in the file pattern to select and load the appropriate TFRecord files.
        - buffer_size: int, size of the buffer around fire masks for spatial splitting before they are considered separate events.
        - do_spatial_split: bool, whether to split the data into regions based on the fire mask
    """
    def dataset_generator():
    
        file_pattern = f"../dataset/*{mode}*.tfrecord"

        dataset = get_dataset(
            file_pattern,
            data_size=64,
            sample_size=64,
            batch_size=100,
            num_in_channels=12,
            compression_type=None,
            clip_and_normalize=False,
            clip_and_rescale=False,
            random_crop=False,
            center_crop=False)

        for inputs, labels in dataset:  # Stream batches
            if do_spatial_split:
                # Split the data into regions based on the fire mask
                fire_mask = inputs[:, :, :, -1]
                regions, y_regions = split_data(inputs, fire_mask, buffer_size, labels)
                for i in range(len(regions)):
                    yield regions[i], y_regions[i]
            else:
                # If not splitting, yield the whole batch
                for i in range(len(inputs)):
                    yield inputs[i], labels[i]

    return dataset_generator

def get_dataset_from_generator(mode = 'train', buffer_size=5, do_spatial_split=True):
    """
    Create a TensorFlow dataset from the dataset generator.
    Parameters, passed to the generator:
        - mode: 'train', 'val', or 'test' to specify the dataset type.
        - buffer_size: int, size of the buffer around fire masks for spatial splitting before they are considered separate events.
        - do_spatial_split: bool, whether to split the data into regions based on the fire mask
    """
    gen = get_dataset_generator(mode = mode, buffer_size=buffer_size, do_spatial_split=do_spatial_split)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 12), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        )
    )
    return dataset
