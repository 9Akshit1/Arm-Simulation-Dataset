import os
import shutil
import re
from pathlib import Path

def get_episode_number(folder_name):
    """Extract episode number from folder name like 'episode_123'"""
    match = re.match(r'episode_(\d+)', folder_name)
    return int(match.group(1)) if match else None

def find_next_available_number(destination_folder, base_name="episode"):
    """Find the next available episode number in the destination folder"""
    existing_numbers = set()
    
    for item in os.listdir(destination_folder):
        if os.path.isdir(os.path.join(destination_folder, item)):
            episode_num = get_episode_number(item)
            if episode_num is not None:
                existing_numbers.add(episode_num)
    
    # Find the smallest available number starting from 1
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    
    return next_num

def copy_episode_folders(source_folder, destination_folder):
    """
    Copy episode folders from source to destination, renaming if conflicts exist
    """
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all episode folders from source
    episode_folders = []
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        if os.path.isdir(item_path) and item.startswith('episode_'):
            episode_folders.append(item)
    
    if not episode_folders:
        print("No episode folders found in source directory.")
        return
    
    # Sort by episode number
    episode_folders.sort(key=lambda x: get_episode_number(x) or 0)
    
    copied_count = 0
    for folder_name in episode_folders:
        source_path = os.path.join(source_folder, folder_name)
        destination_path = os.path.join(destination_folder, folder_name)
        
        # Check if folder already exists in destination
        if os.path.exists(destination_path):
            # Find next available episode number
            next_num = find_next_available_number(destination_folder)
            new_folder_name = f"episode_{next_num}"
            destination_path = os.path.join(destination_folder, new_folder_name)
            
            print(f"Conflict detected: {folder_name} already exists.")
            print(f"Renaming to: {new_folder_name}")
        else:
            new_folder_name = folder_name
            print(f"Copying: {folder_name}")
        
        try:
            # Copy the entire folder
            shutil.copytree(source_path, destination_path)
            copied_count += 1
            print(f"Successfully copied to: {new_folder_name}")
            
        except Exception as e:
            print(f"Error copying {folder_name}: {str(e)}")
    
    print(f"\nCompleted! Copied {copied_count} episode folders.")

# Example usage
if __name__ == "__main__":
    # Set your source and destination paths here
    source_dataset_folder = "dataset"
    destination_dataset_folder = "good_dataset"
    
    print(f"Source folder: {source_dataset_folder}")
    print(f"Destination folder: {destination_dataset_folder}")
    print("-" * 50)
    
    # Check if source folder exists
    if not os.path.exists(source_dataset_folder):
        print(f"Error: Source folder '{source_dataset_folder}' does not exist!")
    else:
        copy_episode_folders(source_dataset_folder, destination_dataset_folder)