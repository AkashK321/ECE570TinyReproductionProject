import os
import urllib.request
import zipfile
import shutil

def download_and_filter_data():
    # Configuration
    url = 'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1'
    zip_filename = 'datasets.zip'
    extract_root = '.'           # Where to unzip (current dir)
    dataset_dir = 'datasets'     # The specific folder name created by the zip
    folder_to_keep = 'zara1'

    try:
        # 1. Download
        print(f"Downloading {zip_filename}...")
        urllib.request.urlretrieve(url, zip_filename)
        print("Download complete.")

        # 2. Extract
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_root)
        print("Extraction complete.")

        # 3. Filter Folders (Remove everything except 'zara1')
        if os.path.exists(dataset_dir):
            print(f"Cleaning up {dataset_dir} (Keeping only '{folder_to_keep}')...")
            
            # List all items in the datasets folder
            for item_name in os.listdir(dataset_dir):
                item_path = os.path.join(dataset_dir, item_name)
                
                # We only care about directories
                if os.path.isdir(item_path):
                    if item_name != folder_to_keep:
                        print(f"  - Removing: {item_name}")
                        shutil.rmtree(item_path)
                    else:
                        print(f"  + Keeping: {item_name}")
        else:
            print(f"Warning: The directory '{dataset_dir}' was not found after extraction.")

        # 4. Clean up zip file
        print(f"Removing temporary file {zip_filename}...")
        if os.path.exists(zip_filename):
            os.remove(zip_filename)

        print("Data setup and cleanup finished successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_filter_data()