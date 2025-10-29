import numpy as np

def transcribe(data : np.ndarray) :
    '''
    Takes time series and transcribes to notes 

    Parameters
    ------------
    data : ndarray 
        An array with every row corresponding to a frame in the video. Rows are populated [left_stick_y_coord, right_stick_y_coord]
    
    '''

    # First, we need to find the global maximum y value
    # the furthest away from the top edge of the frame
    # Should be the y-level of the table

    maxY = 0
    for row in data:
        maxY = max(maxY, row[0], row[1])
    print(maxY)

    # Want to consider distribution of y values and take the top whatever % as hits on the table. 

    

a = np.array([
    [1,1],
    [2,3],
    [4,5]])

transcribe(a)