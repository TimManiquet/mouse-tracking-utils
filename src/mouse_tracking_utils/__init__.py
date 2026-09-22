#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Utility functions to analyse mouse tracking data

Author Tim Maniquet
Created 2026/08/28 09:32:42
'''

#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Utility functions to analyse mouse tracker data

Author Tim Maniquet
Created 2025/03/27 23:19:37
'''

# Imports
import glob
import json
import os
from importlib.resources import files

import numpy as np
import pandas as pd

import time
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from sklearn.manifold import MDS


def progress_bar(n_iterations, current_iteration, n_chunks=5, name=None):
    """Print the progress of a loop."""
    chunks = min(n_chunks, n_iterations) if n_iterations > 0 else 1
    chunk_size = max(n_iterations // chunks, 1)
    name = name or 'iteration'

    if current_iteration == 0 or (current_iteration + 1) % chunk_size == 0 or current_iteration + 1 == n_iterations:
        portion = (current_iteration + 1) / chunk_size
        bar = '*' * int(portion) + '_' * (chunks - int(portion))
        print(f'Progress: [{bar}] {current_iteration + 1}/{n_iterations} {name}s complete')

        if current_iteration + 1 == n_iterations:
            print(f'All {n_iterations} {name}s completed')


def extend_array(input_array, indices, length):
    """
    Extend an array
    ---------------
    Takes in an input array with corresponding indices. Returns
    an array of desired length with the input array values 
    placed at the input indices. Rest of the output array is 
    NaN values.

    Args:
        input_array (np array): input values.
        indices (np array): input indices to place the values.
        length (int): desired length of the output array.

    Returns:
        array: the resulting length-long array with input 
            values placed at the input indices.
    """
    # check if there is something at all
    if len(input_array) > 0:
        # make an empty array with the desired length
        array = np.empty(length)
        array[:] = np.nan # fill it with NaNs
        # add the existing values at their correct spots
        array[indices] = input_array
    # otherwise return an array of nan values
    else:
        array = np.empty(length)
        array[:] = np.nan
    
    return array

def euclidean_distance(point1, point2):
    '''
    Returns the euclidean distance between two points
    '''
    return np.linalg.norm(np.array(point1) - np.array(point2))

def cosine_distance(start, mouse_pos, target_pos):
    """
    Computes the cosine distance between the movement vector (start -> mouse_pos)
    and the ideal direction vector (start -> target_pos).
    """
    # Define vectors
    v = np.array(mouse_pos) - np.array(start)  # Movement vector
    t = np.array(target_pos) - np.array(start)  # Target direction vector

    # Compute norms
    norm_v = np.linalg.norm(v)
    norm_t = np.linalg.norm(t)

    # Avoid division by zero
    if norm_v == 0 or norm_t == 0:
        return 1  # Maximum distance if no movement or zero-length target vector

    # Compute cosine similarity
    cos_theta = np.dot(v, t) / (norm_v * norm_t)

    # Convert to cosine distance
    return 1 - cos_theta

def cosine_distance_vectorized(start, trajectory_points, target_pos):
    """
    Vectorized cosine distance computation.
    
    Args:
        start: Starting position tuple (2,)
        trajectory_points: Array of shape (n_points, 2)
        target_pos: Target position tuple (2,)
    
    Returns:
        Array of shape (n_points,) with cosine distances
    """
    start = np.asarray(start)
    target_pos = np.asarray(target_pos)
    
    # Movement vectors for all points
    v = trajectory_points - start  # (n_points, 2)
    
    # Target direction (constant for all points)
    t = target_pos - start  # (2,)
    
    # Compute norms
    norm_v = np.linalg.norm(v, axis=1)  # (n_points,)
    norm_t = np.linalg.norm(t)  # scalar
    
    # Handle zero cases
    valid_mask = (norm_v > 0) & (norm_t > 0)
    distances = np.ones(len(trajectory_points))
    
    if norm_t > 0:
        cos_theta = np.dot(v[valid_mask], t) / (norm_v[valid_mask] * norm_t)
        distances[valid_mask] = 1 - np.clip(cos_theta, -1, 1)  # clip for numerical stability
    
    return distances

def fill_in_array_fast(input_array, fill_start=False, origin_value=0.0):
    """
    Efficiently fills in NaNs using pandas interpolation.
    Args:
        input_array (np array): the array to fill in.
        fill_start (bool): whether or not to fill in the empty values at the
            start of the array. If False, the array will most likely start 
            with NaN values.
        origin_value (float): if fill_start is True, the first value is set to this
            value and used to fill in the empty values at the start of the
            array.
    """
    s = pd.Series(input_array)

    # If filling the start, set the first value to origin_value
    if fill_start and pd.isna(s.iloc[0]):
        s.iloc[0] = origin_value
    
    # Interpolate internal NaNs linearly
    s_interp = s.interpolate(method='linear', limit_direction='both')

    # If we don't want to fill the leading NaNs, restore them
    if not fill_start:
        first_valid = s.first_valid_index()
        if first_valid is not None:
            s_interp[:first_valid] = np.nan

    # Do not fill trailing NaNs
    last_valid = s.last_valid_index()
    if last_valid is not None and last_valid + 1 < len(s_interp):
        s_interp[last_valid + 1:] = np.nan

    return s_interp.to_numpy()

def find_ems(file, return_demographics = False):
    '''
    Take in a data file and output a EMS number, if any is present
    in the data.
    
    Args:
        file: either a .csv filename or a pd dataframe.
        return_demographics: whether or not to return the whole 
            survey results.
    '''
    # if the input is a filename
    if type(file) == str:
        df = pd.read_csv(file)
    # else if it's a dataframe
    elif isinstance(file, pd.DataFrame):
        df = file.copy()
    # find the demographics survey results
    demographics = df['response'].dropna().iloc[0]    
    # read the output as a dictionary
    demographics = json.loads(demographics)
    # extract the ems code
    ems_code = demographics['EMScode']
    
    if not return_demographics:
        return ems_code
    elif return_demographics:
        ems_code, demographics

def unpack_mouse_data(mouse_data):
    """
    Unpack mouse tracking data
    --------------------------
    This function takes a json-like trial mouse data, and returns
    arrays 'x', 'y', 't' and 'event', extracted from the data
    contained in the 'mouse' data.

    Args:
        mouse : cell of the 'mouse_tracking_data' 
            column.

    Returns:
        df: dataframe containing the four relevant columns.
    """    
    
    mouse_data = json.loads(mouse_data)
    
    x_coord = [data['x'] for data in mouse_data]
    y_coord = [data['y'] for data in mouse_data]
    time = [data['t'] for data in mouse_data]
    event = [data['event'] for data in mouse_data]
    
    return x_coord, y_coord, time, event

def detect_data_files(data_dir, extension = 'csv', criterion = 'ems'):
    
    data_files = [file for file in glob.glob(os.path.join(data_dir, f'*.{extension}')) if criterion in file]
    
    return data_files

def unpack_screen_properties(df, radius_proportion = 0.7):
    
    output = df.copy()
    
    # Turn the response buttons into a clean array
    output['response_buttons'] = output['response_buttons'].apply(json.loads)

    # Turn the screen dimensions into clean dictionaries
    output['screen_dimensions'] = output['screen_dimensions'].apply(json.loads)

    # Extract the circular array radius based on the formula
    output['radius'] = output['screen_dimensions'].apply(lambda x: radius_proportion * x['window_height']/2)

    return output


def collect_data(data_dir, trial_duration, radius_proportion = 0.7, verbose = True):
    
    '''
    This is the function from my 8afc task code
    '''
    
    
    start = time.time()

    data_files = detect_data_files(data_dir)

    if verbose:
        print(f'Processing {len(data_files)} participants...')

    # Declare a empty list of data trials
    dfs = []
    # Loop through data files and extend their content
    for i, data_file in enumerate(data_files):
        data = pd.read_csv(data_file)
        data['ppt_number'] = i+1
        data['ems_code'] = find_ems(data)
        dfs.append(data)
    
    # Merge the dfs together
    df = pd.concat(dfs)

    ## Preprocessing

    # Extract trials from the data
    df = df.loc[(df['phase']=='main_task')&(~pd.isna(df['mask']))]

    # Extract the stimulus names because I was too stupid to record it
    df['image'] = df['stimulus'].apply(lambda x: x.split('main/')[1].split('.png')[0])

    # Extract screen properties
    if verbose:
        print(f' -> Unpacking screen properties...')
    df = unpack_screen_properties(df, radius_proportion = radius_proportion)

    # Unpack the mouse tracking data
    if verbose:
        print(f' -> Unpacking mouse data...')
    df[['x', 'y', 'time', 'event']] = pd.DataFrame(df['mouse_tracking_data'].apply(unpack_mouse_data).tolist(), index=df.index)

    # Standardise the x and y coordinates 
    if verbose:
        print(f' -> Standardising x and y trajectories...')
    df['std_x'] = df.apply(lambda row: np.array(row['x']) / row['screen_dimensions']['window_width'], axis = 1)
    df['std_y'] = df.apply(lambda row: np.array(row['y']) / row['screen_dimensions']['window_height'], axis = 1)

    # Move the mean to 0
    df['std_x'] = df['std_x'] - 0.5
    df['std_y'] = df['std_y'] - 0.5

    # Correct the x coordinates by elongating them slightly
    if verbose:
        print(f' -> Correcting the x trajectories for screen ratio...')
    df['std_x'] = df.apply(lambda row: np.array(row['std_x']) * (radius_proportion / (row['radius']/(row['screen_dimensions']['window_width']/2))), axis = 1)

    # Extend the standardised x and y arrays
    if verbose:
        print(f' -> Extending standardised x and y trajectories...')
    df['ext_std_x'] = df.apply(lambda row: extend_array(row['std_x'], row['time'], trial_duration), axis=1)
    df['ext_std_y'] = df.apply(lambda row: extend_array(row['std_y'], row['time'], trial_duration), axis=1)

    # Fill in the extended arrays by extrapolating the missing values
    if verbose:
        print(f' -> Filling in extended standardised x and y trajectories...')
    df['full_ext_std_x'] = df['ext_std_x'].apply(fill_in_array_fast)
    df['full_ext_std_y'] = df['ext_std_y'].apply(fill_in_array_fast)
    
    # Reset the index
    df.reset_index(inplace=True)
    
    end = time.time()
    
    # Print some final feedback
    if verbose:
        print(f'Done collecting data from {len(df)} trials (n={len(data_files)}, {np.round(end - start)}s elapsed).')
    return df

def preprocess_data(parameters, verbose=True):
    '''
    This is the function from my jocn analysis code
    
    Umbrella function to collect and parse the data from all the participants.
    Returns the aggregated raw data with mouse tracking data unpacked, aligned, 
    standardized and scaled. Saves the resulting dataframe as a pickle file if specified.
    
    Steps:
    1. list all participant files
    2. exclude aborted runs
    3. read all data, add ems and ppt number
    4. concatenate all dataframes and reset index
    5. parse out data files
    6. align x trials based on a reference order of the 2 buttons
    7. standardize coordinates by screen dimensions to account for screen size differences
    8. scale x and y coordinates to a common range (-1, 1 for x; 0, 1 for y)
    9. exclude trials based on mouse data quality criteria (outlier trials based on trajectory)
    10. add a 'image' column with the image filename
    11. save the resulting dataframe as a pickle file if specified
    
    date: 31/10/2025
    '''

    if verbose:
        print('Parsing data …')

    # list all the participants files
    files = glob.glob(os.path.join(parameters['data_dir'], '*.csv'))

    if verbose:
        print(f'Found {len(files)} files.')

    # exclude aborted runs
    files = exclude_files(files)

    if verbose:
        print(f'Keeping {len(files)} files after excluding aborted runs.')

    # read all data, add ems, ppt number and demogaphics
    dfs = []
    for i, data_file in enumerate(files):
        data = pd.read_csv(data_file)
        data['ppt_number'] = i+1
        ems_code, demographics = find_ems(data, return_demographics=True)
        data['ems_code'] = int(ems_code)
        data['handedness'] = demographics['handedness']
        data['gender'] = demographics['gender']
        data['mouse_type'] = demographics['MouseSelection']
        data['age'] = demographics['age']
        
        dfs.append(data)
    
    # concatenate all dataframes and reset index
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    
    if verbose:
        print(f'Parsing out data files …')
    
    df = filter_out_data(df, sort_col=parameters['sort_col'])
    df[['x', 'y', 'time', 'event']] = pd.DataFrame(df['mouse_tracking_data'].apply(unpack_mouse_data).to_list(), index=df.index)
    df = align_x_trials(df, order=parameters['order_reference'])
    
    # standardize coordinates by screen dimensions to account for screen size differences 
    df['std_x'] = df.apply(lambda row: np.array(row['x']) / row['screen_width'], axis = 1)
    df['std_y'] = df.apply(lambda row: np.array(row['y']) / row['screen_height'], axis = 1)
    
    # Move the mean of x to 0
    df['std_x'] = df['std_x'] - 0.5
    # Invert y to put origin at the bottom
    df['std_y'] = 1 - df['std_y']


    # Scale the x coordinates so they span from -1 to 1
    df['scaled_x'] = df['std_x'].apply(lambda arr: (np.array(arr) / parameters['range_x']).tolist())
    # Scale the y coordinates so they span from 0 to 1
    df['scaled_y'] = df['std_y'].apply(lambda arr: ((np.array(arr) - parameters['bottom_y']) / (parameters['top_y'] - parameters['bottom_y'])).tolist())

    n_trials = len(df)
    n_participants = len(df['ppt_number'].unique())
    
    # Remove participants who don't have enough trials left
    ppts_to_remove = []
    for ppt in df['ppt_number'].unique():
        if len(df.loc[df['ppt_number']==ppt]) < parameters['min_nb_trials']:
            ppts_to_remove.append(ppt)
    df = df.loc[~df['ppt_number'].isin(ppts_to_remove)]
    
    df = exclude_trials(df, 'scaled_x', 'scaled_y')

    df['image'] = df['stimulus'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    if verbose:
        print(f'Excluded {n_participants - len(df["ppt_number"].unique())} participants.')
        print(f'Excluded {n_trials - len(df)} trials.')
    if parameters['save_file'] == True:
        outdir = os.path.join(parameters['out_data_dir'], f'{get_current_datetime()}_preprocessed_data.pkl')
        df.to_pickle(outdir)

        if verbose:
            print(f'Parsed data saved as {outdir}.')
    
    return df
    

def exclude_trials_by_distance(
    df:pd.DataFrame,
    origin:tuple,
    min_distance = 0.05,
    max_distance = 10,
    x_col = 'x',
    y_col = 'y',
    verbose = True
    ):
    '''
    Excludes trials that do not end far away enough or too far from a lower and
    upper threshold, respectively.
    Default values consider standardised buttons positions along a circle of 
    radius 0.35, with button size 0.05, with a starting position at 0.0.
    The function will evaluate the euclidean distance between the last data point of
    each trajectory in each trial of the input dataframe, found using the last non-NaN
    values of the x_col and y_col columns, and the origin point given as input.
    
    Args:
        df: input dataframe.
        origin: the origin point from which to compute the euclidean distance.
            In >2-way categorisation tasks where buttons are arranged along a circle, this
            will typically be (0,0)
        min_distance: minimum euclidean distance required between the last point
            of the trajectory and the origin point.
        max_distance: maximum euclidean distance allowed between the last point
            of the trajectory and the origin point.
        x_col: the column name containing the x coordinates.
        y_col: the column name containing the y coordinates.
    Returns:
        output: the cleaned dataframe.
    '''
    # Make a copy of the input
    output = df.copy()
    # List to store indices of rows to drop
    rows_to_drop = []
    # Check that the x and y columns are present
    if (x_col not in output.columns) or (y_col not in output.columns):
        raise ValueError(f'Columns {x_col} and/or {y_col} not found in input dataframe.')
    # Iterate through the rows
    for index, row in output.iterrows():
        # Find the last non-NaN x and y coordinates
        x_values = np.array(row[x_col])
        x_values = x_values[~np.isnan(x_values)]
        if len(x_values) == 0:
            rows_to_drop.append(index)
            continue
        last_x = x_values[-1]
        y_values = np.array(row[y_col])
        y_values = y_values[~np.isnan(y_values)]
        if len(y_values) == 0:
            rows_to_drop.append(index)
            continue
        last_y = y_values[-1]
        
        # Calculate the euclidean distance between the last point and the origin
        distance = euclidean_distance((last_x, last_y), origin)
        # Remove the row if the distance is not within the specified range
        if (distance < min_distance) or (distance > max_distance):
            rows_to_drop.append(index)
        
    # Drop the marked rows
    rows_to_drop = list(set(rows_to_drop)) # remove duplicates
    output.drop(rows_to_drop, inplace=True)
    if verbose:
        print(f'Excluded {len(rows_to_drop)} trials based on distance criteria.')

    return output

def extract_category_distance(df, button_positions, metric, categories=None, ppt_column='ppt_number', verbose=True):
    """
    Extracts average category-wise Euclidean distances for trajectory data 
    from a given dataframe, for all categories and for all images.
    This function computes the distances between trajectory points and predefined 
    button positions for each trial in the dataset. It organizes the results 
    into two structures: one based on categories and another based on images.
    Args:
        df (pd.DataFrame): Input dataframe containing trajectory data. It must 
            include the following columns:
            - 'category': The category label for each trial.
            - 'image': The image identifier for each trial.
            - 'ppt_number': The participant identifier.
            - 'response_buttons': The order of response buttons for each participant.
            - 'full_ext_std_x': The x-coordinates of the trajectory points.
            - 'full_ext_std_y': The y-coordinates of the trajectory points.
        categories (list): A list of category labels to compute distances for. 
            If empty, all unique categories in the dataframe will be used.
        button_positions (list): A list of predefined positions corresponding to 
            the response buttons.
        ppt_column (str, optional): The column name for participant identifiers. 
            Defaults to 'ppt_number'.
        verbose (bool, optional): If True, prints progress feedback during 
            computation. Defaults to True.
    Returns:
        tuple: A tuple containing:
            - category_distances (dict): A nested dictionary where the outer keys 
                are category labels, the inner keys are category labels, and the 
                values are lists of Euclidean distances for each trial.
            - image_distances (dict): A nested dictionary where the outer keys 
                are image identifiers, the inner keys are category labels, and the 
                values are lists of Euclidean distances for each trial.
    Notes:
        - The function assumes that the 'response_buttons' column contains the 
            order of response buttons for each participant, and that the 
            `button_positions` list corresponds to these buttons.
        - The trajectory data in 'full_ext_std_x' and 'full_ext_std_y' columns 
            must be convertible to NumPy arrays.
        - The function uses `np.linalg.norm` to compute Euclidean distances.
    Example:
        category_distances, image_distances = extract_category_distance(
            df=dataframe,
            categories=['cat1', 'cat2', 'cat3'],
            button_positions=[(0, 0), (1, 0), (0, 1)],
            ppt_column='participant_id',
            verbose=True
        )
    """
    
    # If no specific category specified, extract the unique categories
    if not categories:
        categories = df['category'].unique()
    
    # Create an empty category-based structure to hold onto resutls
    category_distances = {category: {cat: [] for cat in categories} for category in categories}
    
    # Create an empty image-based structure to hold onto results
    image_distances = {image: {cat: [] for cat in categories} for image in df['image'].unique()}
    
    # Extract the unique participants from the input dataframe
    ppts = df[ppt_column].unique()
    
    # Print some progress feedback
    if verbose:
        print(f'Extracting category-wise distances for {len(ppts)} participants...')
    
    # Loop over participants to extract the category-wise distances
    for ppt_idx, ppt in enumerate(ppts):
        
        # Print some progress feedback
        if verbose:
            progress_bar(len(ppts), ppt_idx)
        
        # Extract the trials from the current participants
        ppt_data = df.loc[df['ppt_number'] == ppt]

        # For the current participant, get the order of the response buttons
        ppt_response_buttons = ppt_data['response_buttons'].values[0]
        # # Fetch the angle of each category
        # category_angles = {
        #     category: (i / len(categories)) * 2 * np.pi
        #     for i, category in enumerate(ppt_response_buttons)
        # }
        # Fetch the position of each category response button
        std_category_positions = {
            category.replace(' ', ''): position
            for category, position in zip(ppt_response_buttons, button_positions)
        }

        # Precompute category positions for all categories
        category_positions = {
            category: std_category_positions[category]
            for category in categories
        }

        # Precompute full_ext_std_x and full_ext_std_y as arrays for faster access
        ppt_data['full_ext_std_x'] = ppt_data['full_ext_std_x'].apply(np.array)
        ppt_data['full_ext_std_y'] = ppt_data['full_ext_std_y'].apply(np.array)

        # Loop across trials for the current participant
        for _, row in ppt_data.iterrows():
            # Extract the current image and category
            image = row['image']
            image_category = row['category']

            # Precompute trajectory points as a single array
            trajectory_points = np.column_stack((row['full_ext_std_x'], row['full_ext_std_y']))

            # Loop over all categories for the current trial
            for category, category_position in category_positions.items():
                if metric == 'euclidean':
                    # Calculate the euclidean distances for the entire trajectory
                    distances = np.linalg.norm(trajectory_points - category_position, axis=1)
                elif metric == 'cosine':
                    # Calculate the cosine distances for the entire trajectory
                    distances = [cosine_distance((0,0), point, category_position) for point in trajectory_points]

                # Append the results to the image-based results structure
                image_distances[image][category].append(distances)
                # Append the results to the category-based results structure
                category_distances[image_category][category].append(distances)
    
    # Print some progress feedback
    if verbose:
        print(f'Done extracting category-wise distances.')
    
    return category_distances, image_distances


def compute_spatial_distance_weights(
    button_positions, 
    correct_button_idx,
    weighting_scheme='amplify_distant'
):
    """
    Compute spatial distance weights for buttons based on their distance from 
    the correct button. This accounts for the confound that spatially-close 
    buttons are more likely to be reached by accident.
    
    Args:
        button_positions (list): List of (x, y) tuples for each button position.
        correct_button_idx (int): Index of the correct button.
        weighting_scheme (str): How to weight based on spatial distance:
            - 'amplify_distant': Upweight buttons farther from correct button
              (default). Weight = spatial_distance / mean(spatial_distances)
            - 'normalize_chance': Normalize by expected chance distance. Each 
              button's weight is relative to its baseline distance, so buttons
              farther away (with higher chance distance) have lower weight.
            - 'none': No weighting (all weights = 1)
    
    Returns:
        np.ndarray: Array of weights with shape (n_buttons,). Weights are 
            normalized so the mean weight is 1.0.
    """
    n_buttons = len(button_positions)
    correct_pos = np.array(button_positions[correct_button_idx])
    
    # Compute spatial distances from correct button to all buttons
    spatial_distances = np.array([
        euclidean_distance(correct_pos, button_positions[i])
        for i in range(n_buttons)
    ])
    
    if weighting_scheme == 'amplify_distant':
        # Upweight buttons farther away
        # For buttons at distance 0 (correct button), weight = 0
        # For buttons farther, weight = distance / mean_distance
        weights = spatial_distances.copy()
        valid_distances = spatial_distances[spatial_distances > 0]
        if len(valid_distances) > 0:
            mean_distance = np.mean(valid_distances)
            weights[spatial_distances > 0] = spatial_distances[spatial_distances > 0] / mean_distance
        
    elif weighting_scheme == 'normalize_chance':
        # Normalize by expected chance distance.
        # Buttons farther away have higher baseline distance, so their measured
        # distance is less informative about semantic structure. Lower their weight.
        # Weight = 1 / spatial_distance (normalized)
        weights = np.zeros(n_buttons)
        valid_mask = spatial_distances > 0
        if np.any(valid_mask):
            weights[valid_mask] = 1.0 / spatial_distances[valid_mask]
            # Normalize so mean weight is 1.0
            weights[valid_mask] = weights[valid_mask] / np.mean(weights[valid_mask])
    
    elif weighting_scheme == 'none':
        weights = np.ones(n_buttons)
    
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")
    
    return weights


def project_to_equidistant_space(button_positions):
    """
    NOTE: This function is currently not usable due to dimensionality mismatch.
    Trajectories are in 2D but equidistant buttons require 7D space for 8 buttons.
    There is no unique way to transform 2D points to 7D space while preserving
    geometric relationships.
    
    RECOMMENDATION: Use spatial_weighting=True with weighting_scheme='normalize_chance'
    instead, which corrects for spatial confounds without dimensional transformation.
    
    Project button positions into equidistant space using Multidimensional Scaling.
    
    Args:
        button_positions (list): List of (x, y) tuples for button positions.
    
    Returns:
        tuple: (projected_button_positions, transform_func)
    """
    raise NotImplementedError(
        "Equidistant projection is not feasible: trajectories are 2D but "
        "equidistant buttons require 7D space. Use spatial_weighting=True "
        "with weighting_scheme='normalize_chance' instead."
    )


def apply_spatial_weights_to_distances(
    img_cat_distances, 
    button_positions, 
    correct_categories,
    response_buttons,
    categories,
    weighting_scheme='amplify_distant'
):
    """
    Apply spatial distance weights to distance matrices to account for 
    randomized button positions.
    
    Args:
        img_cat_distances (np.ndarray): Distance matrix of shape 
            (n_images, n_categories, n_timepoints).
        button_positions (list): List of (x, y) tuples for button positions.
        correct_categories (np.ndarray): Array of shape (n_images,) containing 
            the correct category index for each image.
        response_buttons (list): List of button category names in order.
        categories (list): List of all category names.
        weighting_scheme (str): Weighting scheme to use:
            - 'amplify_distant': Upweight buttons farther from correct button
            - 'normalize_chance': Normalize by expected chance distance
            - 'none': No weighting
    
    Returns:
        np.ndarray: Weighted distance matrix, same shape as input.
    """
    weighted_distances = img_cat_distances.copy()
    n_images, n_categories, n_timepoints = img_cat_distances.shape
    
    # Create mapping from category name to index
    category_to_idx = {cat: i for i, cat in enumerate(categories)}
    response_button_to_idx = {btn.replace(' ', ''): i for i, btn in enumerate(response_buttons)}
    
    # For each image, compute weights based on its correct category
    for img_idx in range(n_images):
        correct_cat_idx = int(correct_categories[img_idx])
        correct_cat_name = categories[correct_cat_idx]
        
        # Find the button index for this correct category
        correct_button_idx = response_button_to_idx[correct_cat_name]
        
        # Compute weights for this trial
        weights = compute_spatial_distance_weights(
            button_positions, 
            correct_button_idx,
            weighting_scheme=weighting_scheme
        )
        
        # Apply weights: multiply each category's distances by its corresponding weight
        for cat_idx in range(n_categories):
            cat_name = categories[cat_idx]
            button_idx = response_button_to_idx[cat_name]
            weighted_distances[img_idx, cat_idx, :] *= weights[button_idx]
    
    return weighted_distances


def extract_ppt_category_distance(
    df, 
    button_positions,
    metric,
    images=None,
    categories=None,
    trial_duration=1000,
    response_buttons_col='response_buttons',
    x_coord_col='full_ext_std_x',
    y_coord_col='full_ext_std_y',
    spatial_weighting=False,
    weighting_scheme='amplify_distant',
    equidistant_projection=False
    ):
    """
    Extracts category-wise distances for trajectory data from a dataframe.
    
    Computes distances between trajectory points and predefined button positions
    for each trial. Optionally applies spatial distance weighting to account for
    randomized button positions across participants.
    
    Args:
        df (pd.DataFrame): Input dataframe containing trajectory data with columns:
            - 'category': The category label for each trial.
            - 'image': The image identifier for each trial.
            - 'response_buttons': The order of response buttons for each participant.
            - x_coord_col: The x-coordinates of trajectory points.
            - y_coord_col: The y-coordinates of trajectory points.
        button_positions (list): List of (x, y) tuples for button positions.
        metric (str): Distance metric to use ('euclidean' or 'cosine').
        images (list, optional): Images to include. If None, uses all images.
        categories (list, optional): Categories to compute distances for. 
            If None, uses all unique categories.
        trial_duration (int, optional): Expected length of trajectories. Defaults to 1000.
        response_buttons_col (str, optional): Column name for response buttons. 
            Defaults to 'response_buttons'.
        x_coord_col (str, optional): Column name for x coordinates. 
            Defaults to 'full_ext_std_x'.
        y_coord_col (str, optional): Column name for y coordinates. 
            Defaults to 'full_ext_std_y'.
        spatial_weighting (bool, optional): If True, applies spatial distance 
            weighting to account for randomized button positions. Defaults to False.
        weighting_scheme (str, optional): How to weight based on spatial distance:
            - 'amplify_distant': Upweight buttons farther from correct button (default)
            - 'normalize_chance': Normalize by expected chance distance
            - 'none': No weighting
        equidistant_projection (bool, optional): If True, projects both button 
            positions and trajectory points into equidistant space using MDS and 
            Procrustes alignment. This eliminates spatial confounds by ensuring all 
            buttons are equidistant. Defaults to False.
    
    Returns:
        np.ndarray: Distance matrix of shape (n_images, n_categories, n_timepoints).
            Each element [i, j, t] is the distance from the trajectory at timepoint t
            to button j for image i.
    
    Notes:
        - The 'response_buttons' column must be identical across all trials.
        - Spatial weighting uses the physical distance between buttons to adjust
          the measured distances, helping to isolate semantic effects from spatial
          confounds caused by randomized button arrangements.
    
    Example:
        distances = extract_ppt_category_distance(
            df=dataframe,
            button_positions=[(0, 0), (1, 0), (0, 1), (-1, 0)],
            metric='euclidean',
            categories=['cat', 'dog', 'person', 'car'],
            spatial_weighting=True,
            weighting_scheme='amplify_distant'
        )
    """
    data = df.copy()
    
    # If no specific category specified, extract the unique categories
    if not categories:
        categories = data['category'].unique()
    if not images:
        images = data['image'].unique()
    
    # Project to equidistant space if requested
    if equidistant_projection:
        projected_buttons, transform_trajectories = project_to_equidistant_space(button_positions)
        button_positions_to_use = [tuple(p) for p in projected_buttons]
    else:
        transform_trajectories = None
        button_positions_to_use = button_positions
    
    # Initiate a n_categories*n_images*time_points array to hold onto the distances
    img_cat_distances = np.empty((len(images), len(categories), trial_duration))

    # Get the order of the response buttons
    assert all(data[response_buttons_col].values[0] == data[response_buttons_col].values[-1]), "Response buttons order must be the same for all trials."
    response_buttons = data[response_buttons_col].values[0]
    
    # Fetch the position of each category response button
    std_category_positions = {
        category.replace(' ', ''): position
        for category, position in zip(response_buttons, button_positions_to_use)
    }

    # Precompute category positions for all categories
    category_positions = {
        category: std_category_positions[category]
        for category in categories
    }

    # Precompute full_ext_std_x and full_ext_std_y as arrays for faster access
    data[x_coord_col] = data[x_coord_col].apply(np.array)
    data[y_coord_col] = data[y_coord_col].apply(np.array)

    # Loop across trials
    for _, row in data.iterrows():
        # Extract the current image and category
        image = row['image']
        # image_category = row['category']

        # Precompute trajectory points as a single array
        trajectory_points = np.column_stack((row[x_coord_col], row[y_coord_col]))
        
        # Apply equidistant projection transformation if requested
        if transform_trajectories is not None:
            trajectory_points = transform_trajectories(trajectory_points)

        # Loop over all categories for the current trial
        for target_category, category_position in category_positions.items():
            if metric == 'euclidean':
                # Calculate the euclidean distances for the entire trajectory
                distances = np.linalg.norm(trajectory_points - category_position, axis=1)
            elif metric == 'cosine':
                # Calculate the cosine distances for the entire trajectory
                # distances = [cosine_distance((0,0), point, category_position) for point in trajectory_points]
                distances = cosine_distance_vectorized((0, 0), trajectory_points, category_position)

            img_cat_distances[images.index(image), categories.index(target_category), :] = distances

    # Apply spatial weighting if requested
    if spatial_weighting:
        # Create a mapping from category names to indices for weighting
        correct_categories = data['category'].apply(lambda x: categories.index(x)).values
        img_cat_distances = apply_spatial_weights_to_distances(
            img_cat_distances,
            button_positions,
            correct_categories,
            response_buttons,
            categories,
            weighting_scheme=weighting_scheme
        )

    return img_cat_distances



def remove_diag_flatten(matrix):
    '''
    Remove the diagonal of a matrix and return it flattened.
    '''
    return matrix[~(np.identity(matrix.shape[0], dtype=bool))].ravel()




# def temporal_generalisation_correlation(
#     input_matrix, 
#     target_matrix, 
#     image_indices=None,
#     min_valid_samples=2
# ):
#     """
#     Compute temporal generalisation correlation matrix between two data matrices.
    
#     Generates a correlation matrix by comparing input and target matrices across
#     all combinations of timepoints. Useful for decoding analyses and cross-temporal
#     generalisation studies.
    
#     Args:
#         input_matrix (np.ndarray): Input matrix of shape (n_images, n_categories, n_timepoints_input).
#         target_matrix (np.ndarray): Target matrix of shape (n_images, n_categories, n_timepoints_target).
#         image_indices (list, optional): Indices of images to include in analysis. If None, uses all images.
#         min_valid_samples (int, optional): Minimum number of valid samples required for correlation.
#             Defaults to 2.
    
#     Returns:
#         np.ndarray: Correlation matrix of shape (n_timepoints_input, n_timepoints_target) where each
#             element [i, j] is the Pearson correlation between input_matrix[:, :, i] and 
#             target_matrix[:, :, j] (flattened across images and categories).
#     """
#     # Use all images if not specified
#     if image_indices is None:
#         image_indices = np.arange(input_matrix.shape[0])
    
#     # Subset matrices by image indices
#     input_sub = input_matrix[image_indices, :, :]
#     target_sub = target_matrix[image_indices, :, :]
    
#     # Reshape to (n_samples, n_timepoints)
#     n_timepoints_input = input_sub.shape[2]
#     n_timepoints_target = target_sub.shape[2]
    
#     input_flat = input_sub.reshape(-1, n_timepoints_input)
#     target_flat = target_sub.reshape(-1, n_timepoints_target)
    
#     # Precompute validity masks
#     valid_input = ~np.isnan(input_flat)
#     valid_target = ~np.isnan(target_flat)
    
#     # Initialize correlation matrix
#     corr_matrix = np.full((n_timepoints_input, n_timepoints_target), np.nan)
    
#     # Compute correlations for all timepoint pairs
#     for t_in in range(n_timepoints_input):
#         for t_tgt in range(n_timepoints_target):
#             valid = valid_input[:, t_in] & valid_target[:, t_tgt]
            
#             if valid.sum() >= min_valid_samples:
#                 r, _ = pearsonr(input_flat[valid, t_in], target_flat[valid, t_tgt])
#                 corr_matrix[t_in, t_tgt] = r
    
#     return corr_matrix


# def time_resolved_correlation(
#     input_matrix,
#     target_matrix,
#     image_indices=None,
#     input_timepoint=None,
#     min_valid_samples=2
# ):
#     """
#     Compute time-resolved correlation at a specific input timepoint.
    
#     Correlates a single timepoint from input_matrix across all timepoints of
#     target_matrix. Essentially extracts a horizontal line from the temporal
#     generalisation matrix.
    
#     Args:
#         input_matrix (np.ndarray): Input matrix of shape (n_images, n_categories, n_timepoints_input).
#         target_matrix (np.ndarray): Target matrix of shape (n_images, n_categories, n_timepoints_target).
#         image_indices (list, optional): Indices of images to include. If None, uses all images.
#         input_timepoint (int): Index of the input timepoint to correlate from.
#         min_valid_samples (int, optional): Minimum valid samples for correlation. Defaults to 2.
    
#     Returns:
#         np.ndarray: Array of shape (n_timepoints_target,) containing Pearson correlations
#             between input_matrix[:, :, input_timepoint] and each timepoint of target_matrix.
#     """
#     if input_timepoint is None:
#         raise ValueError("input_timepoint must be specified")
    
#     # Use all images if not specified
#     if image_indices is None:
#         image_indices = np.arange(input_matrix.shape[0])
    
#     # Subset matrices by image indices
#     input_sub = input_matrix[image_indices, :, :]
#     target_sub = target_matrix[image_indices, :, :]
    
#     # Extract the single timepoint from input as 2D slice (n_images, n_categories)
#     input_2d = input_sub[:, :, input_timepoint]
    
#     # Flatten to 1D (n_images * n_categories,)
#     input_flat = input_2d.flatten()
#     valid_input = ~np.isnan(input_flat)
    
#     # Reshape target to (n_images * n_categories, n_timepoints_target)
#     n_timepoints_target = target_sub.shape[2]
#     target_flat = target_sub.reshape(-1, n_timepoints_target)
    
#     # Precompute validity mask for target
#     valid_target = ~np.isnan(target_flat)
    
#     # Initialize correlation array
#     correlations = np.full(n_timepoints_target, np.nan)
    
#     # Compute correlations for all target timepoints
#     for t_tgt in range(n_timepoints_target):
#         valid = valid_input & valid_target[:, t_tgt]
        
#         if valid.sum() >= min_valid_samples:
#             r, _ = pearsonr(input_flat[valid], target_flat[valid, t_tgt])
#             correlations[t_tgt] = r
    
#     return correlations


# def scatter_at_timepoint(
#     input_matrix,
#     target_matrix,
#     image_indices=None,
#     input_timepoint=None,
#     target_timepoint=None,
#     category_index=None,
#     return_data=False
# ):
#     """
#     Extract 2D position data for scatter plot at specific timepoints and category.
    
#     Returns values from both matrices at specified timepoints for a given category,
#     useful for visualising individual image points in a scatter plot for a
#     specific category.
    
#     Args:
#         input_matrix (np.ndarray): Input matrix of shape (n_images, n_categories, n_timepoints_input).
#         target_matrix (np.ndarray): Target matrix of shape (n_images, n_categories, n_timepoints_target).
#         image_indices (list, optional): Indices of images to include. If None, uses all images.
#         input_timepoint (int): Index of timepoint to extract from input_matrix.
#         target_timepoint (int): Index of timepoint to extract from target_matrix.
#         category_index (int): Index of category to extract.
#         return_data (bool, optional): If True, returns data and validity mask. Defaults to False.
    
#     Returns:
#         tuple or np.ndarrays: If return_data=False, returns (input_values, target_values).
#             If return_data=True, returns (input_values, target_values, valid_mask).
#             Each value array has shape (n_images,) containing 1D positions for that category.
#     """
#     if input_timepoint is None or target_timepoint is None:
#         raise ValueError("Both input_timepoint and target_timepoint must be specified")
#     if category_index is None:
#         raise ValueError("category_index must be specified")
    
#     # Use all images if not specified
#     if image_indices is None:
#         image_indices = np.arange(input_matrix.shape[0])
    
#     # Subset matrices by image indices
#     input_sub = input_matrix[image_indices, :, :]
#     target_sub = target_matrix[image_indices, :, :]
    
#     # Extract values at specified timepoints and category
#     input_values = input_sub[:, category_index, input_timepoint]
#     target_values = target_sub[:, category_index, target_timepoint]
    
#     # Compute validity mask
#     valid = ~(np.isnan(input_values) | np.isnan(target_values))
    
#     if return_data:
#         return input_values, target_values, valid
#     else:
#         return input_values, target_values


# -----------------------------------------------------------------------------
# Clean mouse-tracking analysis API
# -----------------------------------------------------------------------------
# Legacy code remains above for reference. The functions below provide a generic,
# experiment-agnostic implementation for the two main steps requested in the
# package specification: preprocessing and distance extraction.
# -----------------------------------------------------------------------------


def _as_list(value):
    """Turn a JSON-like object or string into a Python list where possible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed
        return parsed
    return value


def _safe_array(values, dtype=float):
    if values is None:
        return np.array([], dtype=dtype)
    values = _as_list(values)
    if values is None:
        return np.array([], dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    return arr


def _coerce_position_dict(value):
    """Normalise different target layout encodings to {label: (x, y)}."""
    if value is None:
        return {}

    parsed = _as_list(value)
    if isinstance(parsed, dict):
        out = {}
        for key, coord in parsed.items():
            if isinstance(coord, dict):
                if 'x' in coord and 'y' in coord:
                    out[str(key)] = (float(coord['x']), float(coord['y']))
                elif 'center_x' in coord and 'center_y' in coord:
                    out[str(key)] = (float(coord['center_x']), float(coord['center_y']))
                elif 'position' in coord:
                    pos = coord['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        out[str(key)] = (float(pos[0]), float(pos[1]))
                else:
                    out[str(key)] = tuple(float(v) for v in list(coord.values())[:2])
            elif isinstance(coord, (list, tuple, np.ndarray)) and len(coord) >= 2:
                out[str(key)] = (float(coord[0]), float(coord[1]))
        return out

    if isinstance(parsed, list):
        out = {}
        for item in parsed:
            if isinstance(item, dict):
                label = item.get('label') or item.get('name') or item.get('button') or item.get('target')
                pos = item.get('position') or item.get('coords') or item.get('xy') or item.get('center')
                if label is None:
                    continue
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    out[str(label)] = (float(pos[0]), float(pos[1]))
                elif isinstance(item, dict) and 'x' in item and 'y' in item:
                    out[str(label)] = (float(item['x']), float(item['y']))
        return out

    return {}


def _default_button_positions(n_targets, screen_width=1.0, screen_height=1.0, radius=0.35, center_x=0.5, center_y=0.5):
    """Create an arbitrary circular layout when no positions are provided."""
    if n_targets <= 0:
        return {}
    angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    x_positions = center_x + radius * np.cos(angles)
    y_positions = center_y + radius * np.sin(angles)
    return {str(i): (float(x), float(y)) for i, (x, y) in enumerate(zip(x_positions, y_positions))}


def _derive_target_names(df):
    """Infer the generic response-button labels from a dataframe."""
    for key in ['response_buttons', 'button_names', 'target_names', 'targets']:
        if key not in df.columns:
            continue
        values = df.iloc[0][key]
        parsed = _as_list(values)
        if isinstance(parsed, dict):
            return list(parsed.keys())
        if isinstance(parsed, list):
            names = []
            for item in parsed:
                if isinstance(item, dict):
                    label = item.get('label') or item.get('name') or item.get('button') or item.get('target')
                    if label is not None:
                        names.append(str(label))
                else:
                    names.append(str(item))
            if names:
                return names
    return [str(i) for i in range(max(1, df.shape[0]))]


def _infer_target_layout(df, target_names=None, target_positions=None):
    """Create a button layout for a trial-level dataframe."""
    if target_names is None:
        target_names = _derive_target_names(df)

    positions = {}
    if target_positions is not None:
        positions = _coerce_position_dict(target_positions)

    if not positions:
        for key in ['target_positions', 'button_positions', 'mouse_tracking_targets', 'targets', 'response_button_positions']:
            if key in df.columns:
                first = df.iloc[0][key]
                candidate = _coerce_position_dict(first)
                if candidate:
                    positions = candidate
                    break

    if not positions:
        n_targets = max(len(target_names), 1)
        positions = _default_button_positions(n_targets)

    ordered_names = list(target_names)
    if positions:
        ordered_names = [name for name in ordered_names if name in positions] + [name for name in positions.keys() if name not in ordered_names]

    # Ensure all names have a position.
    for name in ordered_names:
        if name not in positions:
            positions[name] = _default_button_positions(len(ordered_names), center_x=0.5, center_y=0.5)[str(len(positions))]

    return ordered_names, {name: positions[name] for name in ordered_names}


def _parse_mouse_tracking_payload(value):
    """Parse mouse tracking dictionaries stored as JSON strings."""
    if value is None:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        payload = value

    if not isinstance(payload, list):
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = np.array([pt['x'] for pt in payload if isinstance(pt, dict) and 'x' in pt], dtype=float)
    y = np.array([pt['y'] for pt in payload if isinstance(pt, dict) and 'y' in pt], dtype=float)
    t = np.array([pt['t'] for pt in payload if isinstance(pt, dict) and 't' in pt], dtype=float)
    event = np.array([pt.get('event', '') for pt in payload if isinstance(pt, dict)], dtype=object)
    return x, y, t, event


def _normalise_trial_columns(df):
    """Standardise a dataframe so it exposes x, y, time, image, and participant columns."""
    output = df.copy()

    if 'x' not in output.columns or 'y' not in output.columns:
        if 'mouse_tracking_data' in output.columns:
            unpacked = output['mouse_tracking_data'].apply(_parse_mouse_tracking_payload)
            output[['x', 'y', 'time', 'event']] = pd.DataFrame(unpacked.tolist(), index=output.index)

    if 'ppt_number' not in output.columns:
        for candidate in ['ppt_id', 'participant', 'participant_id', 'subject', 'ppt']:
            if candidate in output.columns:
                output['ppt_number'] = output[candidate]
                break
        else:
            output['ppt_number'] = np.arange(len(output)) + 1

    if 'repetition' not in output.columns:
        for candidate in ['repetition', 'trial_number', 'trial_index', 'block_number']:
            if candidate in output.columns:
                output['repetition'] = output[candidate]
                break
        else:
            output['repetition'] = 1

    if 'image' not in output.columns:
        for candidate in ['image', 'stimulus', 'stimulus_name', 'filename']:
            if candidate in output.columns:
                output['image'] = output[candidate]
                break
        else:
            output['image'] = 'image_0'

    if 'x' in output.columns and 'y' in output.columns:
        output['x'] = output['x'].apply(lambda xvals: _safe_array(xvals).tolist())
        output['y'] = output['y'].apply(lambda yvals: _safe_array(yvals).tolist())

    if 'time' not in output.columns:
        if 't' in output.columns:
            output['time'] = output['t']
        else:
            output['time'] = output.apply(lambda row: np.arange(len(_safe_array(row.get('x')))), axis=1)

    return output


def _standardise_coordinates(row, x_col='x', y_col='y', screen_width=None, screen_height=None):
    """Standardise mouse coordinates to a common unit space centred on (0, 0)."""
    x = _safe_array(row.get(x_col))
    y = _safe_array(row.get(y_col))

    if len(x) == 0 and len(y) == 0:
        return np.array([]), np.array([])

    width = row.get('screen_width')
    height = row.get('screen_height')
    if width is None and 'screen_dimensions' in row and isinstance(row['screen_dimensions'], dict):
        width = row['screen_dimensions'].get('window_width')
    if width is None and 'screen_dimensions' in row and isinstance(row['screen_dimensions'], str):
        try:
            width = json.loads(row['screen_dimensions']).get('window_width')
        except Exception:
            width = None
    if height is None and 'screen_dimensions' in row and isinstance(row['screen_dimensions'], dict):
        height = row['screen_dimensions'].get('window_height')
    if height is None and 'screen_dimensions' in row and isinstance(row['screen_dimensions'], str):
        try:
            height = json.loads(row['screen_dimensions']).get('window_height')
        except Exception:
            height = None

    if width is None:
        width = screen_width if screen_width is not None else 1.0
    if height is None:
        height = screen_height if screen_height is not None else 1.0

    x_std = np.asarray(x, dtype=float) / float(width) - 0.5
    y_std = np.asarray(y, dtype=float) / float(height) - 0.5

    if row.get('invert_y', False):
        y_std = -y_std

    return x_std, y_std


def _preprocess_mouse_tracking_dataframe(df, *, verbose=True, standardise=True, screen_width=None, screen_height=None, invert_y=False):
    """Generic preprocessing pipeline for row-wise mouse-tracking trial data."""
    output = _normalise_trial_columns(df.copy())

    if standardise:
        output['x_std'] = output.apply(
            lambda row: _standardise_coordinates(row, 'x', 'y', screen_width=screen_width, screen_height=screen_height)[0],
            axis=1,
        )
        output['y_std'] = output.apply(
            lambda row: _standardise_coordinates(row, 'x', 'y', screen_width=screen_width, screen_height=screen_height)[1],
            axis=1,
        )

    return output


def collect_data(data_dir, *, file_pattern='*.csv', verbose=True, **kwargs):
    """Generic entry point for collecting one or several participant CSV files."""
    allowed = {'standardise', 'screen_width', 'screen_height', 'invert_y'}
    clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    if isinstance(data_dir, (list, tuple)):
        files = [str(path) for path in data_dir]
    elif isinstance(data_dir, str) and os.path.isdir(data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    elif isinstance(data_dir, str):
        files = [data_dir]
    else:
        raise TypeError('data_dir must be a folder, a CSV path, or a list of CSV paths.')

    if not files:
        raise FileNotFoundError(f'No files matching {file_pattern!r} were found in {data_dir!r}.')

    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        frame['source_file'] = os.path.basename(file_path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return preprocess_data(combined, verbose=verbose, **clean_kwargs)


def preprocess_data(data, *, verbose=True, **kwargs):
    """Generic preprocessing wrapper supporting folders, file lists, or data frames."""
    allowed = {'standardise', 'screen_width', 'screen_height', 'invert_y'}
    clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    if isinstance(data, dict):
        if 'data_dir' in data:
            return collect_data(data['data_dir'], verbose=verbose, **clean_kwargs)
        if 'data' in data:
            return preprocess_data(data['data'], verbose=verbose, **clean_kwargs)
        return _preprocess_mouse_tracking_dataframe(pd.DataFrame(data), verbose=verbose, **clean_kwargs)

    if isinstance(data, pd.DataFrame):
        return _preprocess_mouse_tracking_dataframe(data, verbose=verbose, **clean_kwargs)

    if isinstance(data, (list, tuple)):
        frames = []
        for item in data:
            if os.path.isfile(item):
                frames.append(pd.read_csv(item))
            elif isinstance(item, pd.DataFrame):
                frames.append(item)
        if not frames:
            raise FileNotFoundError('No valid CSV/dataframe objects were found in the input list.')
        combined = pd.concat(frames, ignore_index=True)
        return _preprocess_mouse_tracking_dataframe(combined, verbose=verbose, **clean_kwargs)

    if isinstance(data, str):
        if os.path.isdir(data):
            return collect_data(data, verbose=verbose, **clean_kwargs)
        if os.path.isfile(data):
            return _preprocess_mouse_tracking_dataframe(pd.read_csv(data), verbose=verbose, **clean_kwargs)

    raise TypeError('data must be a dataframe, CSV path, folder, or list of CSV paths.')


def _resolve_target_metadata(df, target_names=None, target_order=None, target_positions=None):
    """Infer button names, reference order, and positions for a trial dataframe."""
    if target_names is None:
        first_row = df.iloc[0] if len(df) else pd.Series()
        if 'response_buttons' in first_row.index:
            parsed = _as_list(first_row['response_buttons'])
            if isinstance(parsed, list):
                target_names = [str(item) for item in parsed]
            elif isinstance(parsed, dict):
                target_names = list(parsed.keys())
        if target_names is None:
            target_names = [str(i) for i in range(max(1, len(df.columns)))]

    if target_order is not None:
        target_names = [target_names[i] for i in target_order]

    positions = {} if target_positions is None else _coerce_position_dict(target_positions)
    if not positions:
        for key in ['target_positions', 'button_positions', 'mouse_tracking_targets', 'targets', 'response_button_positions']:
            if key in df.columns:
                candidate = _coerce_position_dict(df.iloc[0][key])
                if candidate:
                    positions = candidate
                    break

    if not positions:
        positions = _default_button_positions(len(target_names))

    for name in target_names:
        if name not in positions:
            positions[name] = _default_button_positions(len(target_names))[str(len(positions) % len(target_names))]

    return list(target_names), {name: positions[name] for name in target_names}


def _compute_distance_series(row, target_label, target_position, metric='euclidean', x_col='x', y_col='y', start_x_col=None, start_y_col=None):
    """Return the distance trajectory for a single button/target on a single trial."""
    x = _safe_array(row.get(x_col), dtype=float)
    y = _safe_array(row.get(y_col), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.array([], dtype=float)

    if start_x_col is None:
        start_x = float(x[0])
    else:
        start_x = float(row.get(start_x_col, x[0]))

    if start_y_col is None:
        start_y = float(y[0])
    else:
        start_y = float(row.get(start_y_col, y[0]))

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if metric == 'position':
        if x_col.lower() in {'x', 'x_pos', 'horizontal'}:
            return x_arr
        if y_col.lower() in {'y', 'y_pos', 'vertical'}:
            return y_arr
        if isinstance(row.get(x_col), (list, tuple, np.ndarray)):
            return np.asarray(row.get(x_col), dtype=float)
        if isinstance(row.get(y_col), (list, tuple, np.ndarray)):
            return np.asarray(row.get(y_col), dtype=float)
        return np.asarray(row.get(x_col, x_arr), dtype=float)

    tx, ty = target_position

    if metric == 'euclidean':
        return np.sqrt((x_arr - tx) ** 2 + (y_arr - ty) ** 2)

    if metric == 'cosine':
        v_x = x_arr - start_x
        v_y = y_arr - start_y
        t_x = tx - start_x
        t_y = ty - start_y
        v_norm = np.sqrt(v_x ** 2 + v_y ** 2)
        t_norm = np.sqrt(t_x ** 2 + t_y ** 2)
        numerator = v_x * t_x + v_y * t_y
        denom = v_norm * t_norm
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_similarity = np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=float), where=(denom > 0))
        return 1.0 - np.clip(cos_similarity, -1.0, 1.0)

    raise ValueError(f'Unsupported distance metric: {metric!r}. Supported metrics are euclidean, cosine, and position.')


def extract_distances(
    data,
    *,
    metric='euclidean',
    target_order=None,
    target_names=None,
    target_positions=None,
    x_col='x',
    y_col='y',
    position_col='x',
    ppt_col='ppt_number',
    repetition_col='repetition',
    image_col='image',
    squeeze_singleton=True,
    start_x_col=None,
    start_y_col=None,
):
    """Compute time-resolved distances (or positions) from trajectories to each target.

    Returns an array with one of the following shapes:
      - Euclidean / cosine: (n_ppts, n_repetitions, n_images, n_targets, n_timepoints)
      - Position: (n_ppts, n_repetitions, n_images, n_timepoints)

    If only one repetition, image, or target is present, the singleton dimension is
    removed by default.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = preprocess_data(data)

    df = _normalise_trial_columns(df)

    if ppt_col not in df.columns:
        df[ppt_col] = np.arange(len(df)) + 1
    if repetition_col not in df.columns:
        df[repetition_col] = 1
    if image_col not in df.columns:
        df[image_col] = 'image_0'

    if target_names is None:
        target_names = []
        for _, row in df.iterrows():
            parsed = _as_list(row.get('response_buttons'))
            if isinstance(parsed, list):
                target_names = [str(item) for item in parsed]
                break
            if isinstance(parsed, dict):
                target_names = list(parsed.keys())
                break
        if not target_names:
            target_names = [str(i) for i in range(1, len(df) + 1)]

    default_names, default_layout = _resolve_target_metadata(df, target_names=target_names, target_order=target_order, target_positions=target_positions)
    target_names = default_names
    target_specs = default_layout

    ppt_levels = list(dict.fromkeys(df[ppt_col].astype(str).tolist()))
    rep_levels = list(dict.fromkeys(df[repetition_col].astype(str).tolist()))
    image_levels = list(dict.fromkeys(df[image_col].astype(str).tolist()))

    max_timepoints = max(len(_safe_array(row.get(x_col))) for _, row in df.iterrows())
    if metric == 'position':
        out = np.full((len(ppt_levels), len(rep_levels), len(image_levels), max_timepoints), np.nan, dtype=float)
    else:
        out = np.full((len(ppt_levels), len(rep_levels), len(image_levels), len(target_names), max_timepoints), np.nan, dtype=float)

    for idx, row in df.iterrows():
        p_idx = ppt_levels.index(str(row[ppt_col]))
        r_idx = rep_levels.index(str(row[repetition_col]))
        i_idx = image_levels.index(str(row[image_col]))

        if metric == 'position':
            series = _compute_distance_series(row, None, None, metric='position', x_col=position_col, y_col=position_col, start_x_col=start_x_col, start_y_col=start_y_col)
            if series.size:
                out[p_idx, r_idx, i_idx, :series.size] = series
            continue

        for t_idx, target_label in enumerate(target_names):
            pos = target_specs.get(target_label)
            if pos is None:
                pos = _default_button_positions(len(target_names)).get(str(t_idx), (0.0, 0.0))
            series = _compute_distance_series(row, target_label, pos, metric=metric, x_col=x_col, y_col=y_col, start_x_col=start_x_col, start_y_col=start_y_col)
            if series.size:
                out[p_idx, r_idx, i_idx, t_idx, :series.size] = series

    if squeeze_singleton:
        out = _squeeze_singleton_dims(out)

    return out


def _squeeze_singleton_dims(arr):
    """Remove singleton axes from the leading dimensions of an array."""
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 1 and arr.shape[1] == 1:
        arr = np.take(arr, 0, axis=1)
    while arr.ndim > 2 and arr.shape[2] == 1:
        arr = np.take(arr, 0, axis=2)
    return arr


def load_demo_data(button_count=None):
    """Load lightweight packaged demo data for 2-, 4-, and 8-button tasks."""
    demo_files = {
        2: 'mouse_tracking_demo_2_buttons.csv',
        4: 'mouse_tracking_demo_4_buttons.csv',
        8: 'mouse_tracking_demo_8_buttons.csv',
    }
    demo_data = files(__package__).joinpath('demo_data')

    if button_count is None:
        return {count: pd.read_csv(demo_data.joinpath(path)) for count, path in demo_files.items()}

    if button_count not in demo_files:
        raise ValueError(f'Unsupported demo dataset size {button_count}. Choose one of: {sorted(demo_files.keys())}.')

    return pd.read_csv(demo_data.joinpath(demo_files[button_count]))


# -----------------------------------------------------------------------------
# Demo data generation (kept lightweight and synthetic)
# -----------------------------------------------------------------------------

def _generate_demo_trial(button_count, ppt_id, repetition, image_name, target_names, target_positions, start_x=0.5, start_y=0.5, jitter=0.08):
    """Generate a synthetic example trial with time-resolved mouse positions."""
    n_timepoints = 20
    times = np.linspace(0, 1, n_timepoints)
    x = np.empty(n_timepoints, dtype=float)
    y = np.empty(n_timepoints, dtype=float)

    target_index = (repetition + ppt_id) % button_count
    target_name = target_names[target_index]
    tx, ty = target_positions[target_name]

    for i, t in enumerate(times):
        progress = t ** 1.5
        x[i] = start_x + (tx - start_x) * progress + np.sin((i + 1) * 0.7 + ppt_id) * jitter
        y[i] = start_y + (ty - start_y) * progress + np.cos((i + 1) * 0.9 + repetition) * jitter

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return {
        'ppt_number': ppt_id,
        'repetition': repetition,
        'image': image_name,
        'response_buttons': json.dumps(target_names),
        'target_positions': json.dumps({name: [float(v[0]), float(v[1])] for name, v in target_positions.items()}),
        'x': json.dumps(x.tolist()),
        'y': json.dumps(y.tolist()),
        'time': json.dumps(times.tolist()),
        'start_x': start_x,
        'start_y': start_y,
    }


def generate_demo_data(button_count=2, n_ppts=2, n_images=2, n_repetitions=2):
    """Create lightweight synthetic demo data for a user-facing example package."""
    if button_count not in {2, 4, 8}:
        raise ValueError('button_count must be one of {2, 4, 8}.')

    # Generate a target layout around the screen centre.
    angles = np.linspace(0, 2 * np.pi, button_count, endpoint=False)
    radius = 0.33
    positions = {
        str(i): (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle))
        for i, angle in enumerate(angles)
    }
    button_names = [str(i) for i in range(button_count)]
    trials = []

    for ppt_id in range(1, n_ppts + 1):
        for image_idx in range(n_images):
            image_name = f'image_{image_idx + 1}'
            for repetition in range(1, n_repetitions + 1):
                trials.append(_generate_demo_trial(button_count, ppt_id, repetition, image_name, button_names, positions))

    return pd.DataFrame(trials)


def create_demo_data_files():
    """Persist the packaged demo CSV files in the package directory."""
    base_dir = os.path.join(os.path.dirname(__file__), 'demo_data')
    os.makedirs(base_dir, exist_ok=True)
    for button_count in [2, 4, 8]:
        df = generate_demo_data(button_count=button_count, n_ppts=2, n_images=2, n_repetitions=2)
        outpath = os.path.join(base_dir, f'mouse_tracking_demo_{button_count}_buttons.csv')
        df.to_csv(outpath, index=False)
    return sorted(os.listdir(base_dir))
