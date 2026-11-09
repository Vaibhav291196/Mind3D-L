import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor

from configs.config import (
    DATA_DIR,
    CLIP_FEATURE_FILE,
)


MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_object_features(images, model, processor):
    """
    Extract CLIP features for all views of an object.

    Args:
        images: (N, V, H, W, C)

    Returns:
        view_features: (N, V, 512)
        object_features: (N, 512)
    """

    num_objects, num_views = images.shape[:2]

    all_view_features = []

    for start in range(0, num_objects * num_views, BATCH_SIZE):

        end = min(start + BATCH_SIZE, num_objects * num_views)

        flat_indices = range(start, end)

        image_batch = []

        for idx in flat_indices:
            obj_idx = idx // num_views
            view_idx = idx % num_views

            image = images[obj_idx, view_idx]

            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            image_batch.append(Image.fromarray(image))

        inputs = processor(
            images=image_batch,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )

            pooled_features = vision_outputs.pooler_output

            features = model.visual_projection(
                pooled_features
            )

            features = F.normalize(features, dim=-1)

        all_view_features.append(
            features.cpu().numpy()
        )

        print(
            f"Processed {end}/{num_objects * num_views} views",
            end="\r"
        )

    all_view_features = np.concatenate(
        all_view_features,
        axis=0
    )

    view_features = all_view_features.reshape(
        num_objects,
        num_views,
        -1
    )

    object_features = view_features.mean(axis=1)

    norms = np.linalg.norm(
        object_features,
        axis=1,
        keepdims=True
    )

    object_features = object_features / np.maximum(norms, 1e-8)

    return view_features, object_features


def main():

    print("Device:", DEVICE)

    print("Loading CLIP model...")

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    model.eval()

    processor = CLIPImageProcessor.from_pretrained(
        MODEL_NAME
    )

    results = {}

    for split in ["train", "val"]:

        file_path = DATA_DIR / f"{split}.npz"

        print(f"\nLoading {split} dataset...")

        data = np.load(file_path)

        images = data["obj_images"]

        print("Images:", images.shape)

        view_features, object_features = extract_object_features(
            images,
            model,
            processor
        )

        results[f"{split}_view_features"] = view_features
        results[f"{split}_object_features"] = object_features

        print("\nView features:", view_features.shape)
        print("Object features:", object_features.shape)

    CLIP_FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        CLIP_FEATURE_FILE,
        **results
    )

    print("\nCLIP features saved to:")
    print(CLIP_FEATURE_FILE)

    print("\nExtraction completed successfully.")


if __name__ == "__main__":
    main()
