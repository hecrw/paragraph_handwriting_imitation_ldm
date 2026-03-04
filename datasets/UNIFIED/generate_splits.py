"""
Generate train/val/test splits by writer for the UNIFIED Arabic dataset.
Split ratio: 80/10/10 by writer (so val/test writers are unseen during training).
Also generates writer_to_id.json mapping string writer IDs to integer IDs.
"""
import json
import os
import random

SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_path = os.path.join(script_dir, "annotations_768.jsonl")

    # Load all annotations
    samples = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Collect writers and their samples
    writer_samples = {}
    for s in samples:
        wid = s["writer_id"]
        if wid not in writer_samples:
            writer_samples[wid] = []
        writer_samples[wid].append(s["id"])

    writers = sorted(writer_samples.keys())
    print(f"Total writers: {len(writers)}, Total samples: {len(samples)}")

    # Shuffle and split writers
    random.seed(SEED)
    random.shuffle(writers)

    n_train = int(len(writers) * SPLIT_RATIO[0])
    n_val = int(len(writers) * SPLIT_RATIO[1])

    train_writers = writers[:n_train]
    val_writers = writers[n_train:n_train + n_val]
    test_writers = writers[n_train + n_val:]

    print(f"Train writers: {len(train_writers)}, Val writers: {len(val_writers)}, Test writers: {len(test_writers)}")

    # Build split sample lists
    splits = {"train": [], "val": [], "test": []}
    for w in train_writers:
        splits["train"].extend(writer_samples[w])
    for w in val_writers:
        splits["val"].extend(writer_samples[w])
    for w in test_writers:
        splits["test"].extend(writer_samples[w])

    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} samples")

    # Write uttlist files
    for split_name, sample_ids in splits.items():
        path = os.path.join(script_dir, f"{split_name}.uttlist")
        with open(path, "w", encoding="utf-8") as f:
            for sid in sorted(sample_ids):
                f.write(sid + "\n")
        print(f"Wrote {path}")

    # Write writer_to_id.json (all writers, sorted, mapped to 0..N-1)
    all_writers_sorted = sorted(writer_samples.keys())
    writer_to_id = {w: i for i, w in enumerate(all_writers_sorted)}
    id_path = os.path.join(script_dir, "writer_to_id.json")
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(writer_to_id, f, indent=2)
    print(f"Wrote {id_path} ({len(writer_to_id)} writers)")


if __name__ == "__main__":
    main()
