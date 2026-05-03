import argparse
import json
import torch
from typing import Optional, Tuple
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

# Load CLIP model and processor once.
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor embedding to unit length."""
    return embedding / torch.norm(embedding, dim=-1, keepdim=True)


def get_text_embedding(text: str) -> torch.Tensor:
    """Convert text into a normalized CLIP text embedding."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    text_embedding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
    return _normalize_embedding(text_embedding)


def get_image_embedding(image_path: str) -> torch.Tensor:
    """Convert an image file into a normalized CLIP image embedding."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    image_embedding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
    return _normalize_embedding(image_embedding)


def fuse_embeddings(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    method: str = "concat",
    weight_a: float = 0.7,
    weight_b: float = 0.3,
) -> torch.Tensor:
    """Fuse two embeddings into a single vector.

    Supported methods:
    - "concat": concatenate and normalize
    - "mean": elementwise mean and normalize
    - "sum": elementwise sum and normalize
    - "weighted": weighted sum and normalize
    """
    if method == "concat":
        fused = torch.cat([embedding_a, embedding_b], dim=-1)
    elif method == "mean":
        fused = (embedding_a + embedding_b) / 2
    elif method == "sum":
        fused = embedding_a + embedding_b
    elif method == "weighted":
        if not 0 <= weight_a <= 1 or not 0 <= weight_b <= 1:
            raise ValueError("Weights must be between 0 and 1")
        fused = embedding_a * weight_a + embedding_b * weight_b
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    return _normalize_embedding(fused)


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    embedding_a = _normalize_embedding(embedding_a)
    embedding_b = _normalize_embedding(embedding_b)
    return float(torch.nn.functional.cosine_similarity(embedding_a, embedding_b, dim=-1).item())


def load_category_embeddings(path: str = "category_embeddings.json") -> dict:
    """Load precomputed category embeddings from JSON and return tensor values."""
    with open(path, "r", encoding="utf-8") as f:
        raw_emb = json.load(f)

    category_embeddings = {}
    for category, subcats in raw_emb.items():
        category_embeddings[category] = {}
        for subcategory, vector in subcats.items():
            tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            category_embeddings[category][subcategory] = _normalize_embedding(tensor)

    return category_embeddings


def find_best_subcategory_match(fused_embedding: torch.Tensor, category_embeddings: dict) -> dict:
    """Return the best matching category/subcategory and its cosine similarity."""
    best_match = {
        "category": None,
        "subCategory": None,
        "score": -1.0,
    }

    for category, subcats in category_embeddings.items():
        for subcategory, emb in subcats.items():
            score = cosine_similarity(fused_embedding, emb)
            if score > best_match["score"]:
                best_match.update({
                    "category": category,
                    "subCategory": subcategory,
                    "score": score,
                })

    return best_match


def embed_text_and_image(text: str, image_path: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return normalized text and optional image embeddings for a text/image pair."""
    text_emb = get_text_embedding(text)
    image_emb = get_image_embedding(image_path) if image_path else None
    return text_emb, image_emb


def get_query_embedding(
    text: str,
    image_path: Optional[str] = None,
    method: str = "weighted",
    weight_text: float = 0.7,
    weight_image: float = 0.3,
) -> torch.Tensor:
    """Return a query embedding, fusing text and image only when an image is provided."""
    text_emb = get_text_embedding(text)
    if image_path is None:
        return text_emb

    image_emb = get_image_embedding(image_path)
    return fuse_embeddings(
        text_emb,
        image_emb,
        method=method,
        weight_a=weight_text,
        weight_b=weight_image,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=False)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--fusion",
        type=str,
        default="weighted",
        choices=["concat", "mean", "sum", "weighted"],
    )
    parser.add_argument("--text_weight", type=float, default=0.7)
    parser.add_argument("--image_weight", type=float, default=0.3)
    parser.add_argument("--category_embeddings", type=str, default="category_embeddings.json")
    args = parser.parse_args()

    query_emb = get_query_embedding(
        text=args.text,
        image_path=args.img_path,
        method=args.fusion,
        weight_text=args.text_weight,
        weight_image=args.image_weight,
    )
    print("query_emb shape:", query_emb.shape)

    category_embeddings = load_category_embeddings(args.category_embeddings)
    best_match = find_best_subcategory_match(query_emb, category_embeddings)

    print("Best category match:", best_match)



