import numpy as np
import matplotlib.pyplot as plt
import math
from music21 import stream, note, tempo, meter

def transcribe(data : np.ndarray, numerator : int, denominator : int) -> np.ndarray:
    '''
    Takes time series and transcribes to notes 

    Parameters
    ------------
    data : ndarray 
        A 1-D array of floats, denoting the time stamp of every stick hit throughout the video
    numerator : int
        The top value in the time signature
    denominator : int
        The bottom value in the time signature
    '''
    # Sort the time series, just in case
    data = np.sort(data)

    # Player should play one measure's worth of initialization notes at the beginning of the video
    # Use the initialization measure to get timing information
    measure_length = (data[1] - data[0]) * denominator

    BPM = (int)((numerator / measure_length) * 60)
    T_beat = 60.0 / BPM
    TIME_SIG = numerator + '/' + denominator
    # I think a reasonable granularity to use for rhythm search is 2^(log2(denominator) + 2)
    # So 3/4 would be searching at a granularity of 2^(log2(4) + 2) = 2^4 = 16th notes
    # I am capping the granularity at 16th notes for simplicity sake
    SUBDIVISIONS_PER_BEAT = min(16, pow(2, (np.log2(denominator) + 2)))
    grid_unit = T_beat / SUBDIVISIONS_PER_BEAT

    # Drop the first measure 
    data = data[numerator:]

    grid_positions = data / grid_unit
    quantized_positions = np.round(grid_positions).astype(int)
    unique_quantized_positions = np.unique(quantized_positions)
    unique_quantized_positions = unique_quantized_positions[unique_quantized_positions >= 0]

    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure()

    score.insert(tempo.MetronomeMark(number=BPM))
    measure.append(meter.TimeSignature(TIME_SIG))
    part.append(measure)

    # Iterate through the quantized indices to create musical notes
    for index in unique_quantized_positions:
        
        granularity = 1/SUBDIVISIONS_PER_BEAT
        offset_in_beats = index * granularity
        
        drum_note = note.Note('C4', type='16th') 
        drum_note.midi = 38 # Standard MIDI for Snare Drum
        
        # Insert the note at the calculated offset (Bar | Beat | Subdivision)
        part.insert(offset_in_beats, drum_note)

    score.insert(0, part)
    score.show()
