import os
import shutil
import glob

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def move_videos(label_file, source_dir, destination_dir, video_extension='.mp4', is_test=False):
    """
    Move videos to the destination directory based on the label file.

    Args:
        label_file (str): Path to the label file.
        source_dir (str): Directory where all .mp4 files are initially stored.
        destination_dir (str): Directory to move the .mp4 files into.
        video_extension (str): Extension of the video files. Defaults to '.mp4'.
        is_test (bool): Flag indicating if the split is 'test'. Determines label file format.
    """
    with open(label_file, 'r') as f:
        for line in f:
            if is_test:
                # Test label files have format: video_id;label_string
                parts = line.strip().split(';')
                if len(parts) == 2:
                    video_id = parts[0]
                else:
                    print(f"Invalid line format in {label_file}: {line.strip()}")
                    continue
            else:
                # Train and Validation label files have format: video_id label_id
                parts = line.strip().split()
                if len(parts) >= 1:
                    video_id = parts[0]
                else:
                    print(f"Invalid line format in {label_file}: {line.strip()}")
                    continue

            video_filename = f"{video_id}{video_extension}"
            source_path = os.path.join(source_dir, video_filename)
            dest_path = os.path.join(destination_dir, video_filename)

            if os.path.exists(source_path):
                shutil.move(source_path, dest_path)
                print(f"Moved {video_filename} to {destination_dir}")
            else:
                print(f"Video file not found: {source_path}")

def copy_labels_json(source_labels_json, destination_dir):
    """Copy labels.json to the destination directory."""
    dest_labels_json = os.path.join(destination_dir, 'labels.json')
    shutil.copy(source_labels_json, dest_labels_json)
    print(f"Copied labels.json to {destination_dir}")

def main():
    # Define base directory
    base_dir = os.path.expanduser('~/ssv2/')
    
    # Define paths
    labels_dir = os.path.join(base_dir, 'labels')
    original_labels_json = os.path.join(labels_dir, 'labels.json')

    # Verify that the original labels.json exists
    if not os.path.exists(original_labels_json):
        print(f"Original labels.json not found at {original_labels_json}. Exiting.")
        return

    # Define split configurations
    splits = {
        'train': {
            'label_file': os.path.join(labels_dir, 'train.txt'),
            'destination_dir': os.path.join(base_dir, 'train'),
            'is_test': False,
        },
        'validation': {
            'label_file': os.path.join(labels_dir, 'validation.txt'),
            'destination_dir': os.path.join(base_dir, 'validation'),
            'is_test': False,
        },
        'test': {
            'label_file': os.path.join(labels_dir, 'test-answers.csv'),
            'destination_dir': os.path.join(base_dir, 'test'),
            'is_test': True,
        },
    }

    # Define the source directory containing all .mp4 files
    # Assuming all videos are initially in the base_dir
    source_video_dir = base_dir

    # Iterate over each split and organize data
    for split_name, details in splits.items():
        print(f"\nProcessing split: {split_name}")
        
        # Create split directory
        create_directory(details['destination_dir'])
        
        # Move videos based on the label file
        move_videos(
            label_file=details['label_file'],
            source_dir=source_video_dir,
            destination_dir=details['destination_dir'],
            video_extension='.mp4',
            is_test=details['is_test']
        )
        
        # Copy labels.json into the split directory
        copy_labels_json(original_labels_json, details['destination_dir'])
    
    print("\nData organization complete!")

if __name__ == "__main__":
    main()
