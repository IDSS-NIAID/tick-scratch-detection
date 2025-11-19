import cv2
import numpy as np
import os
from multiprocessing import Pool
import argparse
from pathlib import Path
import glob

def frameDifference(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
    """Calculate the difference between two frames to detect motion."""
    f1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    f2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    f_diff = cv2.subtract(f1_gray, f2_gray)
    f_diff = cv2.medianBlur(f_diff, 3)

    msk = cv2.adaptiveThreshold(f_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 3)
    
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(msk)

    msk_filtered = np.zeros_like(labels)

    min_size = 9*9 # may need to be adjusted later
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_size:
            msk_filtered[labels == i] = 255

    return msk_filtered

def process_video(args):
    video_path, output_folder, save_with_overlay = args
    
    # Create output subfolder for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get total frame count
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = int(cap.get(cv2.CAP_PROP_FPS))
    
    print('Frames:', total_frames, 'FPS:', fps)
    
    # Read first frame
    ret, frame_prev = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}")
        cap.release()
        return
    
    frame_count = 0
    motion_frame_count = 0
    
    while True:
        ret, frame_current = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame pair using frameDifference
        f_diff = frameDifference(frame_prev, frame_current)
        
        # Check for motion
        if np.amax(f_diff) > 0:
            motion_frame_count += 1
            pixel_count = np.sum(f_diff) // 255
            # save actual frame number so we know the location of the frame in the video 
            output_filename = f"frame-{frame_count:08d}.jpg"
            output_path = os.path.join(video_output_path, output_filename)
            
            if save_with_overlay:
                f_overlay = frame_current.copy()
                f_overlay[:,:,2][f_diff > 0] = 255  # Red overlay on motion areas
                cv2.imwrite(output_path, f_overlay)
            else:
                cv2.imwrite(output_path, frame_current)
        
        # Update previous frame
        frame_prev = frame_current.copy()
    
    cap.release()
    print(f"Processed {video_path}: {motion_frame_count} motion frames detected out of {frame_count} total frames")

def process_videos_in_folder(input_folder, output_folder, save_with_overlay=False):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all video files
    # for now only process MP4 files
    video_extensions = ('.mp4')
    video_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(video_extensions)
    ]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    # Prepare arguments for parallel processing
    process_args = [(video_file, output_folder, save_with_overlay) for video_file in video_files]
    
    # get number of available cores, reserve one for the main thread
    num_cores = len(os.sched_getaffinity(0)) - 1
    num_cores = 1 if num_cores < 1 else num_cores
    
    print(f"Using {num_cores} cores")
    
    # Process videos in parallel
    with Pool(processes=num_cores) as pool:
        pool.map(process_video, process_args)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process videos to extract motion frames')
    parser.add_argument('input_folder', help='Path to folder containing input videos')
    parser.add_argument('output_folder', help='Path to folder for output motion frames')
    parser.add_argument('--overlay', action='store_true', 
                       help='Save frames with motion overlay (red highlights)')
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_folder = str(Path(args.input_folder).resolve())
    output_folder = str(Path(args.output_folder).resolve())
    
    print(f"Processing videos from {input_folder}")
    print(f"Saving motion frames to {output_folder}")
    print(f"Overlay mode: {'enabled' if args.overlay else 'disabled'}")
    
    # Process all videos
    process_videos_in_folder(input_folder, output_folder, save_with_overlay=args.overlay)
    print("Done.")

if __name__ == "__main__":
    main()
