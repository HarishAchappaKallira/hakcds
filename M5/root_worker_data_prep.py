from mpi4py import MPI # Importing mpi4py package from MPI module
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP # Importing Decimal, ROUND_HALF_UP functions from the decimal package
import psutil

#num_threads_machine = psutil.cpu_count(logical=True)
num_threads_machine = 4

#bring all functions in to one fil
#load data
FILENAME = "/Users/achappa/devhak/cds/M5/PowerPlantData.csv" #local file path

def loadData(filename):
    # Loading the dataset with first row as the header and the new column names as suggested
    data = pd.read_csv(filename, header=0,names=['Ambient Temperature','Exhaust Vaccum','Ambient Pressure','Relative Humidity','Energy Output'])
    return data
# Calling the function loadData and storing the dataframe in a variable named df

def standardize_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_standardized = df.copy()

    # Get only numeric columns
    numeric_columns = df.select_dtypes('float64').columns

    # Standardize each numeric column
    for column in numeric_columns:
        mean = df_standardized[column].mean()
        std = df_standardized[column].std()

        # Avoid division by zero
        if std > 0:
            df_standardized[column] = (df_standardized[column] - mean) / std
        else:
            # If standard deviation is 0, just center the data
            df_standardized[column] = df_standardized[column] - mean

    return df_standardized

def prepare_power_plant_data(df):
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Seggregate features and target
    feature_cols = df_copy.columns[:-1]
    target_col = df_copy.columns[-1]

    # Create feature matrix X and target vector y
    X = df_copy[feature_cols]
    y = df_copy[target_col]

    return X, y

#split data
def train_test_split(features, targets, train_size=0.7, random_state=None):
    # Convert to numpy arrays if needed
    if isinstance(features, pd.DataFrame):
        X = features.values
    else:
        X = features
    if isinstance(targets, (pd.Series, pd.DataFrame)):
        y = targets.values
    else:
        y = targets

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    # Calculate split index
    split_idx = int(train_size * X.shape[0])

    # Split the data
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

    return X_train, X_test, y_train, y_test

#divide data ######
def dividing_data(x_train, y_train, size_of_workers):
    # Size of the slice
    slice_for_each_worker = int(Decimal(x_train.shape[0]/size_of_workers).quantize(Decimal('1.'), rounding = ROUND_HALF_UP))
    print('Slice of data for each worker: {}'.format(slice_for_each_worker))
    X = np.array(x_train)
    y = np.array(y_train)
    n_samples = X.shape[0]
    slices = []  
    print('slice for each worker',slice_for_each_worker)

    for i in range(size_of_workers):
        start = i * slice_for_each_worker
        # For the last worker, take all remaining samples
        end = (i + 1) * slice_for_each_worker if i != size_of_workers - 1 else n_samples
        X_slice = X[start:end]
        y_slice = y[start:end]
        slices.append((X_slice, y_slice))
    return slices
###########
# Calculating the coeffients
def estimate_coefficients(X_ec, y_ec):
    # Ensure X is a numpy array
    if isinstance(X_ec, pd.DataFrame):
        X_ec = X_ec.values
    if isinstance(y_ec, (pd.Series, pd.DataFrame)):
        y_ec = y_ec.values

    # Add intercept column (column of ones)
    X_intercept_ec = np.c_[np.ones(X_ec.shape[0]), X_ec]

    # Compute coefficients using the Normal Equation
    beta = np.linalg.inv(X_intercept_ec.T @ X_intercept_ec) @ X_intercept_ec.T @ y_ec
    return beta

# defining a fit function
def fit(X_fit, y_fit):
    # Call the estimate_coefficients function
    beta = estimate_coefficients(X_fit, y_fit)
    # First value is intercept, rest are coefficients
    intercept = beta[0]
    coefficients = beta[1:]
    return intercept, coefficients

def rmse(y_actual, y_pred):
    y_true = np.array(y_actual)
    y_pred = np.array(y_pred)
    mse = np.mean((y_actual - y_pred) ** 2)
    return np.sqrt(mse)


# Defining a function
def main():
    comm = MPI.COMM_WORLD   # Creating a communicator
    rank = comm.Get_rank()  # number of the process running the code
    size = comm.Get_size()  # total number of processes running

    slices = None

    if rank == 0:
        # Divide the data among the workers using the dividing_data function above              
        df = loadData(FILENAME)
        df_standardized = standardize_data(df)
        X, y = prepare_power_plant_data(df_standardized) # Create Feature and target from standardized data   
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42) #split to test_train_splot
        slices = dividing_data(X_train, y_train, num_threads_machine)
    
    # Scatter the slices to all worker processes
    received_data = comm.scatter(slices, root=0)

    #cluster the data and send to each thread
    #for input, do y-pred based on distance from the cluster center

    #do work
    #get X, y from tuple received data 
    X_slice, y_slice = received_data
    # Fit model on the slice
    intercept, coefficients = fit(pd.DataFrame(X_slice), y_slice)
    print(f"Worker {rank}: Intercept: {intercept}, Coefficients: {coefficients}")

    # Displaying the result
    comm.Barrier()  # Synchronize all processes

    #gather the slices 
    # Gather the slices from all workers to the root process
    gathered_data = comm.gather((intercept,coefficients), root=0)
    
    if rank == 0:
        # Root process: combine the slices
        #calculate the average of the intercept
        avg_intercept = np.mean([intercept for intercept, coefficients in gathered_data])
        print(f"Root Worker: Average Intercept: {avg_intercept}")

        #calulate the average of the coefficients
        avg_coefficients = np.mean([coefficients for intercept, coefficients in gathered_data], axis=0)
        print(f"Root Worker: Average Coefficients: {avg_coefficients}")

        # Calculate RMSE for the test set
        y_pred_test = intercept + np.dot(X_test, avg_coefficients)
        test_rmse = rmse(y_test, y_pred_test)
        print("Test RMSE:", test_rmse)


main()
