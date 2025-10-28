import os, glob
from PIL import Image
import imagehash
from tqdm import tqdm
from collections import defaultdict

# ------------------------------
# Helper: List all images recursively
# ------------------------------
def list_images(root):
    files = []
    for ext in (".jpg", ".jpeg", ".png"):
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return files

# ------------------------------
# Helper: Count images per class folder
# ------------------------------
def count_by_class(root):
    counts = defaultdict(int)
    for cls in ["normal", "violence", "weaponized"]:
        for ext in (".jpg", ".jpeg", ".png"):
            files = glob.glob(os.path.join(root, "**", cls, f"*{ext}"), recursive=True)
            counts[cls] += len(files)
    return counts

# ------------------------------
# Helper: Compute perceptual hashes
# ------------------------------
def build_phash_index(paths):
    index = {}
    for p in tqdm(paths, desc=f"Hashing {len(paths)} images"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                index[p] = imagehash.phash(im)
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")
    return index

# ------------------------------
# Helper: Compare hashes between two datasets
# ------------------------------
def compare_hashes(index_a, index_b, max_dist=5):
    matches = []
    for pa, ha in tqdm(index_a.items(), desc="Comparing A ↔ B"):
        for pb, hb in index_b.items():
            dist = ha - hb
            if dist == 0:
                matches.append((pa, pb, dist, "EXACT"))
            elif dist <= max_dist:
                matches.append((pa, pb, dist, "SIMILAR"))
    return matches


# ------------------------------
# MAIN SCRIPT
# ------------------------------
if __name__ == "__main__":
    smart_city_root = "data/smart_city-processed_image"
    rwf_root = "data/rwf2000-processed_image"

    # Count by class
    sc_counts = count_by_class(smart_city_root)
    rwf_counts = count_by_class(rwf_root)

    print("📊 Dataset Summary")
    print(f"Smart City:")
    for cls, n in sc_counts.items():
        print(f"  - {cls}: {n} images")
    print(f"RWF-2000:")
    for cls, n in rwf_counts.items():
        print(f"  - {cls}: {n} images")

    # List all images (all categories)
    smart_imgs = list_images(smart_city_root)
    rwf_imgs = list_images(rwf_root)

    print(f"\n🖼️ Total Smart City images: {len(smart_imgs)}")
    print(f"🖼️ Total RWF-2000 images:   {len(rwf_imgs)}")

    # Build perceptual hash index for both datasets
    smart_idx = build_phash_index(smart_imgs)
    rwf_idx = build_phash_index(rwf_imgs)

    # Compare across datasets
    matches = compare_hashes(smart_idx, rwf_idx, max_dist=5)

    # Separate exact and similar
    exact = [m for m in matches if m[3] == "EXACT"]
    similar = [m for m in matches if m[3] == "SIMILAR"]

    print("\n✅ Comparison complete.")
    print(f"Exact duplicates found:   {len(exact)}")
    print(f"Visually similar matches: {len(similar)}")

    if exact:
        print("\n--- 🔁 Exact duplicates ---")
        for a, b, d, t in exact[:10]:
            print(f"{a}  <->  {b}")
        if len(exact) > 10:
            print(f"... ({len(exact)-10} more)")

    if similar:
        print("\n--- 🟡 Similar images (distance ≤5) ---")
        for a, b, d, t in similar[:10]:
            print(f"d={d} | {a}  <->  {b}")
        if len(similar) > 10:
            print(f"... ({len(similar)-10} more)")
