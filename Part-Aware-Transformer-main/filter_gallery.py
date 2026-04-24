import os
import csv
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = "/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026"
QUERY_CLASSES_CSV = os.path.join(DATASET_DIR, "query_classes.csv")
TEST_CLASSES_CSV  = os.path.join(DATASET_DIR, "test_classes.csv")
QF_PATH = "./qf.npy"
GF_PATH = "./gf.npy"
OUTPUT_CSV = "./submission_filtered.csv"
TOP_K = 100

# ── Load class labels ────────────────────────────────────────────────────────
def load_classes(csv_path):
    classes = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            classes.append(row[2])  # Class column
    return classes

query_classes = load_classes(QUERY_CLASSES_CSV)
test_classes  = load_classes(TEST_CLASSES_CSV)

print(f"Query images: {len(query_classes)}")
print(f"Gallery images: {len(test_classes)}")
print(f"Unique classes: {set(query_classes)}")

# ── Load features ────────────────────────────────────────────────────────────
qf = np.load(QF_PATH)  # (num_query, feat_dim)
gf = np.load(GF_PATH)  # (num_gallery, feat_dim)

print(f"Query features shape: {qf.shape}")
print(f"Gallery features shape: {gf.shape}")

assert len(query_classes) == qf.shape[0], \
    f"Query class count {len(query_classes)} != feature count {qf.shape[0]}"
assert len(test_classes) == gf.shape[0], \
    f"Gallery class count {len(test_classes)} != feature count {gf.shape[0]}"

# ── Class-filtered retrieval ─────────────────────────────────────────────────
# Full distance matrix
q_g_dist = np.dot(qf, np.transpose(gf))  # higher = more similar (cosine, normalized)

num_query   = qf.shape[0]
num_gallery = gf.shape[0]

filtered_indices = []

for i in range(num_query):
    q_class = query_classes[i]

    # Build a mask: True for gallery items of the same class
    same_class_mask = np.array([c == q_class for c in test_classes])
    same_class_idx  = np.where(same_class_mask)[0]
    diff_class_idx  = np.where(~same_class_mask)[0]

    # Sort same-class gallery by similarity (descending → argsort on negated)
    same_class_scores = q_g_dist[i, same_class_idx]
    sorted_same = same_class_idx[np.argsort(-same_class_scores)]

    # Fill remaining slots with different-class gallery (sorted by similarity)
    diff_class_scores = q_g_dist[i, diff_class_idx]
    sorted_diff = diff_class_idx[np.argsort(-diff_class_scores)]

    # Concatenate: same-class first, then different-class
    ranked = np.concatenate([sorted_same, sorted_diff])[:TOP_K]
    filtered_indices.append(ranked)

filtered_indices = np.array(filtered_indices)  # (num_query, TOP_K)

# ── Write submission CSV ─────────────────────────────────────────────────────
lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, num_query + 1)]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['imageName', 'Corresponding Indexes'])
    for img_name, ranked in zip(lista_nombres, filtered_indices):
        # +1 because Kaggle expects 1-indexed gallery
        track_str = ' '.join(map(str, ranked + 1))
        writer.writerow([img_name, track_str])

print(f"\nSaved filtered submission to: {OUTPUT_CSV}")
print(f"Rows written: {len(lista_nombres)}")

# ── Stats ────────────────────────────────────────────────────────────────────
from collections import Counter
class_counts = Counter(query_classes)
print("\nQuery class distribution:")
for cls, cnt in sorted(class_counts.items()):
    gallery_cnt = sum(1 for c in test_classes if c == cls)
    print(f"  {cls}: {cnt} queries, {gallery_cnt} gallery items")
