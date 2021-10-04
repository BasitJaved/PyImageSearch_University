import time

def benchmark(datasetGen, numsteps):
    # start timer
    start = time.time()

    # loop over the provided number of steps
    for i in range(0, numsteps):
        # get the next batch of data
        (images, labels) = next(datasetGen)

    # stop timer
    end = time.time()

    return (end - start)