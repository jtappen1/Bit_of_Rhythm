import cv2
import time
import os

"""
NOTE:  This script is useful for collecting data. This is what I used to decide which frames I wanted to save and label.
"""

def record_and_playback(duration_seconds=5, fps=30):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FPS, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Recording for {duration_seconds} seconds at {fps} FPS...")
    print("Press 'q' to stop recording early")
    
    frames = []
    start_time = time.time()
    frame_delay = 1.0 / fps
    
    while (time.time() - start_time) < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frames.append(frame.copy())
        
        cv2.putText(frame, "RECORDING", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Time: {int(time.time() - start_time)}s", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Webcam Recording', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    print(f"Recording complete! Captured {len(frames)} frames")
    print("\nPlayback Controls:")
    print("  SPACE - Next frame")
    print("  'b'   - Previous frame")
    print("  'p'   - Auto-play/pause")
    print("  'a'   - Save current frame")
    print("  'q'   - Exit playback")
    
    i = 0
    paused = True

    # Create folder for saved frames
    save_folder = f"saved_frames _{time.time()}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    while i < len(frames):
        frame = frames[i].copy()
        
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(frame, f"PLAYBACK - {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i+1}/{len(frames)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Next | B=Back | P=Play | A=Save | Q=Quit", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Webcam Recording', frame)
        
        wait_time = int(1000/fps) if not paused else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            i = min(i + 1, len(frames) - 1)
            paused = True
        elif key == ord('b'):
            i = max(i - 1, 0)
            paused = True
        elif key == ord('p'):
            paused = not paused
        elif key == ord('a'):
            # Save current frame
            filename = os.path.join(save_folder, f"frame_{i+1}.png")
            cv2.imwrite(filename, frames[i])
            print(f"Saved {filename}")
            paused = True

        if not paused:
            i += 1
            if i >= len(frames):
                i = len(frames) - 1
                paused = True
    
    cv2.destroyAllWindows()
    print("Playback complete!")

if __name__ == "__main__":
    record_and_playback(duration_seconds=5, fps=30)