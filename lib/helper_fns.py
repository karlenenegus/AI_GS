from pathlib import Path
import json

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def resolve_file_path(file_path, base_dir):
    """Resolve file path, checking absolute path first, then relative to base_dir."""
    file_path_obj = Path(file_path)
    if file_path_obj.exists():
        return str(file_path_obj.resolve())
    elif (Path(base_dir) / file_path_obj).exists():
        return str((Path(base_dir) / file_path_obj).resolve())
    else:
        raise ValueError(f"File {file_path} does not exist (checked: {file_path} and {Path(base_dir) / file_path})")