import cv2
import os

# --- Configuration ---
# 1. REPLACE THIS WITH THE ACTUAL PATH TO YOUR VIDEO FILE
VIDEO_FILE_PATH = 'Video Examples\\train\\far-light-surface-fast-2.mov' 
# 2. Directory where the extracted frames will be saved
OUTPUT_DIR = 'data\\training_frames'
# 3. Frame extraction interval (e.g., 15 means every 15th frame)
FRAME_SKIP_INTERVAL = 15 

def extract_frames(video_path, output_dir, interval):
    """
    Extracts frames from a video at a specified interval and saves them 
    to the output directory.

    Requires the 'opencv-python' library: pip install opencv-python
    """
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Frame counter
    frame_count = 0
    # Number of frames successfully saved
    saved_count = 0

    print(f"Starting frame extraction (saving every {interval}th frame)...")
    
    # Loop through the video frames
    while True:
        # Read the next frame
        success, frame = cap.read()

        # If reading was not successful, we have reached the end of the video
        if not success:
            break

        # Check if the current frame number is a multiple of the interval
        if frame_count % interval == 0:
            # Construct the output filename
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            
            # Save the frame as a JPEG image
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            # print(f"Saved {frame_filename}")

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved to '{output_dir}': {saved_count}")
    print("Extraction complete.")


if __name__ == "__main__":
    # Ensure you replace the placeholder path before running!
    extract_frames(VIDEO_FILE_PATH, OUTPUT_DIR, FRAME_SKIP_INTERVAL)
