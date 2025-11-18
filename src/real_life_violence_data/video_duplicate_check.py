import os, hashlib
from pathlib import Path

def md5(file_path, chunk_size=8192):
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def collect_hashes(root_dir):
    hashes = {}
    total = 0
    for cls in ['Violence', 'NonViolence']:
        folder = Path(root_dir) / cls
        if not folder.exists():
            print(f"⚠️ Folder not found: {folder}")
            continue
        count = 0
        for f in folder.glob("*.mp4"):
            hashes[f] = md5(f)
            count += 1
        print(f"{cls}: {count} videos read from {folder}")
        total += count
    print(f"Total videos scanned: {total}")
    return hashes

def find_duplicates(hashes):
    hash_to_files = {}
    for path, h in hashes.items():
        hash_to_files.setdefault(h, []).append(path)
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    print(f"Found duplicates: {len(duplicates)}")
    return duplicates

def remove_duplicates(duplicates, dry_run=True):
    deleted = 0
    for h, files in duplicates.items():
        # keep the first one, delete the rest
        keep = files[0]
        to_delete = files[1:]
        for f in to_delete:
            if dry_run:
                print(f"[Dry Run] Would delete: {f}")
            else:
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
        print(f"Kept: {keep}")
    if not dry_run:
        print(f"✅ Deleted {deleted} duplicate files.")

# ---------- run ----------
root = "./data/Real_life_Violence_Dataset"
hashes = collect_hashes(root)
duplicates = find_duplicates(hashes)
remove_duplicates(duplicates, dry_run=True)  # change to False after verifying
