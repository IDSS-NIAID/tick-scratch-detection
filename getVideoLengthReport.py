import argparse
import cv2
import pandas as pd
from pathlib import Path

def get_video_length_hours(video_path: Path) -> float | None:
    """
    Calculates the length of a single video file in hours.
    Returns the duration in hours, or None if an error occurs.
    """
    try:
        # Open the video file
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            print(f"Warning: Could not open video file: {video_path}")
            return None

        # Get video properties
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)

        # Ensure properties are valid to avoid division by zero
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
            duration_hours = duration_seconds / 3600
            return duration_hours
        else:
            print(f"Warning: Could not get valid frame count or FPS for {video_path}. Skipping.")
            return None

    except Exception as e:
        print(f"Error processing file {video_path}: {e}")
        return None
    finally:
        if 'video' in locals() and video.isOpened():
            video.release()

def process_folder(root_folder_path: str):
    """
    Recursively finds all MP4 files in a folder, calculates their lengths,
    and saves the results to an Excel file in that same folder.
    """
    root_path = Path(root_folder_path)
    if not root_path.is_dir():
        print(f"Error: The specified folder does not exist: {root_folder_path}")
        return

    print(f"Scanning for MP4 videos in '{root_path}'...")

    # Use rglob to find all .mp4 files recursively
    video_files = list(root_path.rglob('*.mp4'))

    if not video_files:
        print("No MP4 files found in the specified folder or its subdirectories.")
        return

    print(f"Found {len(video_files)} MP4 files. Now processing each one...")

    # List to hold the data for our spreadsheet
    report_data = []

    # Iterate over all found video files
    for video_file in video_files:
        print(f"  - Analyzing: {video_file.name}")
        length_hours = get_video_length_hours(video_file)

        # If the length was calculated successfully, add it to our report
        if length_hours is not None:
            report_data.append({
                'Video Full Path': str(video_file.resolve()), # .resolve() gets the full absolute path
                'Length (hours)': length_hours
            })

    if not report_data:
        print("Could not gather any video length data. Exiting.")
        return

    # --- Save the results to an Excel file ---
    print("\nAll videos processed. Saving results to Excel...")

    # Create a pandas DataFrame from our list of dictionaries
    df = pd.DataFrame(report_data)

    # Define the output file path
    output_excel_path = root_path / 'video_lengths_report.xlsx'

    try:
        # Save the DataFrame to an Excel file
        # index=False prevents pandas from writing row indices to the spreadsheet
        df.to_excel(output_excel_path, index=False)
        print(f"\nSuccess! Report saved to: {output_excel_path}")
    except Exception as e:
        print(f"\nError: Could not save the Excel file. Reason: {e}")


if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="A script to recursively find all MP4 videos in a folder, calculate their length in hours, and report the results in an Excel spreadsheet."
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="The full path to the root folder you want to scan."
    )

    args = parser.parse_args()

    # Run the main function with the provided folder path
    process_folder(args.root_folder)
