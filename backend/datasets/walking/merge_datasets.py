import os
import shutil
import random
import uuid

def merge_datasets(folders, output_folder, ratios=None):
    """
    Merge datasets from multiple folders into a new folder with specified ratios.

    :param folders: List of paths to the folders.
    :param output_folder: Path to the output folder.
    :param ratios: List of ratios for each folder (must sum to 1.0). If None, equal distribution is used.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set default ratios if not provided
    if ratios is None:
        ratios = [1.0 / len(folders)] * len(folders)
    
    if len(folders) != len(ratios):
        raise ValueError("Number of folders must match number of ratios")
    
    if abs(sum(ratios) - 1.0) > 0.0001:
        raise ValueError("Ratios must sum to 1.0")

    # Get all files from all folders
    all_files = []
    for folder in folders:
        files = os.listdir(folder)
        random.shuffle(files)  # Shuffle for randomness
        all_files.append(files)

    # Calculate number of files to take from each folder
    total_available = sum(len(files) for files in all_files)
    
    # Calculate the limiting factor based on ratios and available files
    limiting_factors = [len(files) / ratio if ratio > 0 else float('inf') 
                       for files, ratio in zip(all_files, ratios)]
    total_files = int(min(total_available, min(limiting_factors)))
    
    # Calculate number of files to take from each folder
    num_files = [int(total_files * ratio) for ratio in ratios]
    
    # Adjust the last element to ensure the sum matches total_files
    num_files[-1] = total_files - sum(num_files[:-1])
    
    # Ensure all counts are integers
    num_files = [int(count) for count in num_files]
    
    # Select and copy files
    for i, (folder, files, count) in enumerate(zip(folders, all_files, num_files)):
        selected_files = files[:count]
        for file in selected_files:
            unique_name = f"folder{i+1}_{uuid.uuid4().hex}_{file}"
            shutil.copy(os.path.join(folder, file), os.path.join(output_folder, unique_name))
    
    # Print summary
    summary = ", ".join(f"{count} files from {folder}" for folder, count in zip(folders, num_files))
    print(f"Merged {summary} into {output_folder}.")

def randomize_and_rename_files(directory):
    """
    Randomly reorder and rename all files in the specified directory.

    :param directory: Path to the directory containing files to be renamed.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Randomly shuffle files
    random.shuffle(files)

    # Rename files sequentially
    for index, file in enumerate(files):
        old_path = os.path.join(directory, file)
        new_name = f"random_file_{index + 1}.csv"
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)

    print(f"Randomly reordered and renamed {len(files)} files in {directory}.")

if __name__ == "__main__":
    # Example usage for three folders
    folder1 = "backend/datasets/walking/splited/band_accel_stable_stand"
    folder2 = "backend/datasets/walking/splited/0"
    folder3 = "backend/datasets/walking/splited/band_accel_stable_tilt"  # 添加第三个文件夹
    output_folder = "backend/datasets/walking/splited/merged_dataset"

    # 合并三个文件夹，比例为 0.5, 0.25, 0.25
    merge_datasets([folder1, folder2, folder3], output_folder, ratios=[0.33, 0.33, 0.34])

    target_directory = "backend/datasets/walking/splited/merged_dataset"
    randomize_and_rename_files(target_directory)
