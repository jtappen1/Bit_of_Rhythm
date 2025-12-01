import numpy as np
import matplotlib.pyplot as plt
import math
from music21 import stream, note, tempo, meter
import verovio
import os

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

    BPM = int((numerator / measure_length) * 60)
    T_beat = 60.0 / BPM
    TIME_SIG = str(numerator) + '/' + str(denominator)
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

    # Create score and part
    score = stream.Score()
    part = stream.Part()
    
    # Add tempo and time signature to the part
    part.append(tempo.MetronomeMark(number=BPM))
    part.append(meter.TimeSignature(TIME_SIG))

    # Calculate quarter note length per measure
    quarter_notes_per_measure = (4.0 / denominator) * numerator
    
    # Create a dictionary to organize notes by measure
    measures_dict = {}
    
    for index in unique_quantized_positions:
        granularity = 1/SUBDIVISIONS_PER_BEAT
        offset_in_beats = index * granularity
        
        # Calculate which measure this note belongs to
        measure_number = int(offset_in_beats / quarter_notes_per_measure)
        offset_in_measure = offset_in_beats % quarter_notes_per_measure
        
        if measure_number not in measures_dict:
            measures_dict[measure_number] = []
        
        measures_dict[measure_number].append(offset_in_measure)
    
    # Create measures with notes
    for measure_num in sorted(measures_dict.keys()):
        m = stream.Measure(number=measure_num + 1)
        
        # Add notes to this measure
        for offset in measures_dict[measure_num]:
            drum_note = note.Note('C4', quarterLength=0.25)  # 16th note
            drum_note.notehead = 'x'  # Use 'x' notehead for drums
            m.insert(offset, drum_note)
        
        # Fill the rest of the measure with rests if needed
        m.makeRests(fillGaps=True, inPlace=True)
        part.append(m)
    
    score.insert(0, part)
    
    # Write to temporary file first
    temp_file = 'temp_musicxml.xml'
    score.write('musicxml', fp=temp_file)
    
    # Read the file content as string
    with open(temp_file, 'r', encoding='utf-8') as f:
        musicxml_string = f.read()
    
    # Clean up temp file
    os.remove(temp_file)
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 2100,
        "pageHeight": 2970,
        "scale": 40,
        "adjustPageHeight": True,
        "footer": "none",
        "header": "none"
    })
    tk.loadData(musicxml_string)

    svg_output = tk.renderToSVG(1)
    with open('music_render.html', 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
        <html>
        <head>
            <title>Music Transcription</title>
            <style>
                body {{ 
                    margin: 20px; 
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Drum Transcription</h1>
                <p>BPM: {BPM} | Time Signature: {TIME_SIG} | Notes: {len(unique_quantized_positions)}</p>
                {svg_output}
            </div>
        </body>
        </html>
        ''')
    