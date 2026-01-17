import os
import shutil
import math

def split_configs_into_batches():
    print("--- Config Batch Splitter ---\n")

    # 1. Get Source Path (The 'configs' folder)
    source_path = input("Enter the full path to your source 'configs' folder: ").strip().strip('"').strip("'")
    
    if not os.path.isdir(source_path):
        print(f"Error: The directory '{source_path}' does not exist.")
        return

    # 2. Get Destination Path
    dest_base_path = input("Enter the full path where you want to save the batch folders: ").strip().strip('"').strip("'")
    
    if not os.path.exists(dest_base_path):
        create_dest = input(f"The destination '{dest_base_path}' does not exist. Create it? (y/n): ").lower()
        if create_dest == 'y':
            os.makedirs(dest_base_path)
        else:
            print("Operation cancelled.")
            return

    # 3. Get Number of Batches
    try:
        num_batches = int(input("Enter the number of batches (folders) you want: "))
        if num_batches <= 0:
            print("Error: Number of batches must be greater than 0.")
            return
    except ValueError:
        print("Error: Please enter a valid number.")
        return

    # 4. Get all files and sort them ascendingly
    # We filter to ensure we only get files, not nested folders
    all_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    all_files.sort() # Ascending sort

    total_files = len(all_files)
    
    if total_files == 0:
        print("Error: The source folder is empty.")
        return

    print(f"\nFound {total_files} files. Splitting into {num_batches} batches...")

    # 5. Calculate Split Logic
    # base_size is the minimum files per batch
    # remainder is how many batches get 1 extra file
    base_size = total_files // num_batches
    remainder = total_files % num_batches

    current_file_idx = 0

    for batch_num in range(1, num_batches + 1):
        # Determine the size of this specific batch
        # If the batch number (0-indexed logic) is less than remainder, it gets an extra file
        current_batch_size = base_size + (1 if (batch_num - 1) < remainder else 0)

        # Create the batch folder
        batch_folder_name = f"batch_{batch_num}"
        batch_folder_path = os.path.join(dest_base_path, batch_folder_name)
        
        if not os.path.exists(batch_folder_path):
            os.makedirs(batch_folder_path)

        # Slice the file list for this batch
        batch_files = all_files[current_file_idx : current_file_idx + current_batch_size]
        
        # Copy files to the new batch folder
        for filename in batch_files:
            src_file = os.path.join(source_path, filename)
            dst_file = os.path.join(batch_folder_path, filename)
            shutil.copy2(src_file, dst_file) # copy2 preserves metadata
        
        print(f"  -> Created '{batch_folder_name}' with {len(batch_files)} files.")
        
        # Update the index for the next iteration
        current_file_idx += current_batch_size

    print("\nProcessing complete! All files have been split successfully.")

if __name__ == "__main__":
    split_configs_into_batches()