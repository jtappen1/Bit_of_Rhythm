import cv2
import os
import time
from tkinter import Tk, filedialog

def select_video_file():
    """Open a file dialog to select a video file."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    root.destroy()
    return file_path

def browse_video(video_path, fps_override=None):
    """Allows browsing through video frames and saving selected ones."""
    if not video_path or not os.path.exists(video_path):
        print("No video selected or file not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Opened {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print("Controls:")
    print("  SPACE - Next frame")
    print("  B     - Previous frame")
    print("  P     - Play/Pause")
    print("  S     - Save current frame")
    print("  Q     - Quit")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join("saved_images", f"{base_name}_images")
    os.makedirs(save_dir, exist_ok=True)

    frame_idx = 0
    paused = True

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(frame, f"{status} | Frame {frame_idx+1}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Next | B=Back | P=Play | S=Save | Q=Quit",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Video Browser", frame)

        wait_time = int(1000 / fps) if not paused else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Next frame
            frame_idx = min(frame_idx + 1, total_frames - 1)
            paused = True
        elif key == ord('b'):  # Previous frame
            frame_idx = max(frame_idx - 1, 0)
            paused = True
        elif key == ord('p'):  # Play/pause toggle
            paused = not paused
        elif key == ord('s'):  # Save current frame
            filename = os.path.join(save_dir, f"frame_{frame_idx+1}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        elif not paused:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved frames are in: {save_dir}")

if __name__ == "__main__":
    video_path = select_video_file()
    browse_video(video_path)
