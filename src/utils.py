import os
import shutil

def clean_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"🧹 Deleted: {directory_path}")
        except OSError as e:
            print(f"⚠️ Can't delete {directory_path}: {e}")
    else:
        print(f"ℹ️ Directory is not exist, creating new one: {directory_path}")