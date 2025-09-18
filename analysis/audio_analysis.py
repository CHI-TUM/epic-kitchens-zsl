import os
import argparse
from pydub import AudioSegment

def calculate_duration_stats(folder_path):
    total_duration = 0
    min_duration = float('inf')
    max_duration = 0
    num_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            
            # Load the audio file using pydub
            audio = AudioSegment.from_mp3(file_path)

            # Get the duration in seconds
            duration_seconds = len(audio) / 1000.0

            # Update statistics
            total_duration += duration_seconds
            min_duration = min(min_duration, duration_seconds)
            max_duration = max(max_duration, duration_seconds)
            num_files += 1

    # Calculate mean duration
    mean_duration = total_duration / num_files if num_files > 0 else 0

    return min_duration, max_duration, mean_duration, total_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Audio Analysis")
    parser.add_argument(
        "--path", 
        required=True, 
        type=str, 
        help="The path to your folder containing the MP3 files."
    )
    args = parser.parse_args()
    
    folder_path = args.path
    min_duration, max_duration, mean_duration, total_duration = calculate_duration_stats(folder_path)

    print(f"Minimum Duration: {min_duration:.2f} seconds")
    print(f"Maximum Duration: {max_duration:.2f} seconds")
    print(f"Mean Duration: {mean_duration:.2f} seconds")
    print(f"Total Duration: {total_duration:.2f} seconds")
