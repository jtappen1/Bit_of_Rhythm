import numpy as np
import matplotlib.pyplot as plt
import math

def transcribe(data : np.ndarray, numerator : int, denominator : int) -> np.ndarray:
    '''
    Takes time series and transcribes to notes 

    Parameters
    ------------
    data : ndarray 
        A 1-D array of *numpy.datetime64* objects denoting the time stamp of every stick hit throughout the video
    numerator : int
        The top value in the time signature
    denominator : int
        The bottom value in the time signature
    '''
    # Sort the time series, just in case
    data = np.sort(data)

    # Player should play one measure's worth of initialization notes at the beginning of the video
    # Use the initialization measure to get timing information
    measure_length = data[numerator] - data[0]

    # Drop the first measure 
    data = data[numerator:]
    num_measures = math.ceil((data[-1] - data[0]) / measure_length)

    notes_per_measure = np.empty(shape=(num_measures), dtype=object)
    for measure in range(num_measures):
        # Get all of the hits in the time range for this measure. Stored in current_measure
        # TODO: CHANGE THIS LOGIC TO ACCOMODATE THE TIMESTAMP DATASTRUCTURE
        bottom_range = data[0] + measure * measure_length
        top_range = bottom_range + measure_length
        mask = (data >= bottom_range) & (data < top_range)
        current_measure = data[mask]

        # I think a reasonable granularity to use for rhythm search is 2^(log2(denominator) + 2)
        # So 6/8 would be searching at a granularity of 2^(log2(8) + 2) = 2^5 = 32nd notes
        # Likewise, anything __/4 time signature would be searching 16th notes
        granularity = pow(2, (np.log2(denominator) + 2))
        num_timestamps = granularity + 1

        bottom_range_int = bottom_range.astype('datetime64[ns]').astype(np.int64)
        top_range_int = top_range.astype('datetime64[ns]').astype(np.int64)

        time_array_int = np.linspace(
            bottom_range_int, 
            top_range_int, 
            num=num_timestamps,
            dtype=np.int64
        )
        # convert the integer array back to datetime64[ns]
        time_array = time_array_int.astype('datetime64[ns]')

        current_measure_int = current_measure.astype('datetime64[ns]').astype(np.int64)
        bins = np.digitize(current_measure_int, time_array_int)

        # 1 if any element falls in bin [time_array[i], time_array[i+1])
        binary_array = np.zeros(len(time_array) - 1, dtype=int)
        for i in range(len(time_array) - 1):
            binary_array[i] = np.any((bins == i + 1))
        
        notes_per_measure[measure] = binary_array
        return notes_per_measure

transcribe(np.array([1,2,3,4,5,6]),2,1)