from mpi4py import MPI # Importing mpi4py package from MPI module
# Define a function
def main():
    # creating the communicator
    comm = MPI.COMM_WORLD
    # number of the process running the code i.e rank
    rank = comm.Get_rank()
    # total number of processes running i.e size
    size = comm.Get_size()
    # Displaying the rank and size of a communicator
    print("rank is {} and size is {}".format(rank,size))

# invoke the function
main()
