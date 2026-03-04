import os
import json
from PIL import Image
from tqdm import tqdm
from src.data.utils.alphabet import Alphabet


def preload_all_unified(root, split, alphabet=None, mode="L", size=(768, 768),
                        max_samples=-1, **kwargs):
    if alphabet is None:
        alphabet = Alphabet(dataset="UNIFIED", mode="both")

    # Load writer_to_id mapping
    writer_map_path = os.path.join(root, "writer_to_id.json")
    with open(writer_map_path, "r", encoding="utf-8") as f:
        writer_to_id = json.load(f)

    # Load sample IDs for this split
    uttlist_path = os.path.join(root, f"{split}.uttlist")
    with open(uttlist_path, "r", encoding="utf-8") as f:
        split_ids = set(line.strip() for line in f if line.strip())

    # Load annotations and filter by split
    annotations_path = os.path.join(root, "annotations_768.jsonl")
    annotations = {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry["id"] in split_ids:
                annotations[entry["id"]] = entry

    sample_names = sorted(annotations.keys())
    if 0 < max_samples < len(sample_names):
        sample_names = sample_names[:max_samples]

    images = {}
    meta_data = {}

    for s in tqdm(sample_names, desc=f"Loading UNIFIED {split}"):
        entry = annotations[s]

        # Load image (already 768x768 grayscale)
        img_path = os.path.join(root, "images_768", f"{s}.png")
        img = Image.open(img_path).convert(mode)
        images[s] = img

        writer_int = writer_to_id[entry["writer_id"]]
        meta_data[s] = {
            "text": entry["text"],
            "writer": writer_int,
            "text_logits": alphabet.string_to_logits(entry["text"]),
        }

    return sample_names, meta_data, images
