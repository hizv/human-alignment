"""Download and load official SavedModels with a consistent inference API."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

FAMILIES = {"dino": "DINOv2-B", "siglip": "SigLIP-B"}
VARIANTS = {"base": "base_model", "alignet": "alignet", "unalignet": "untransformed"}
INDEX_URL = "https://storage.googleapis.com/alignet/models/index.html"


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def checkpoint_name(family: str, variant: str) -> str:
    return f"{FAMILIES[family]}-{VARIANTS[variant]}"


def resolve_url(name: str) -> str:
    """Read the publisher's index rather than assume an archive URL."""
    with urlopen(INDEX_URL, timeout=120) as response:
        parser = _Links()
        parser.feed(response.read().decode("utf-8"))
    matches = []
    for link in parser.links:
        url = urljoin(INDEX_URL, link)
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if filename.lower() == f"{name}.tar.gz".lower():
            matches.append(url)
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive for {name}, found {matches}. Check {INDEX_URL}")
    return matches[0]


def ensure_checkpoint(family: str, variant: str, cache_dir="models", *, download=False) -> Path:
    """Return a cached SavedModel, downloading only when explicitly requested."""
    name = checkpoint_name(family, variant)
    cache = Path(cache_dir).expanduser().resolve()
    target = cache / name
    if (target / "saved_model.pb").is_file():
        return target
    if not download:
        raise FileNotFoundError(f"Missing {target}/saved_model.pb; pass --download to fetch it.")
    if target.exists():
        raise RuntimeError(f"Incomplete checkpoint directory: {target}. Move it aside before retrying.")
    cache.mkdir(parents=True, exist_ok=True)
    url = resolve_url(name)
    print(f"Downloading {name} from {url}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as temporary:
        temporary = Path(temporary)
        archive = temporary / "checkpoint.tar.gz"
        with urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        unpacked = temporary / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(unpacked, filter="data")
        roots = list(unpacked.rglob("saved_model.pb"))
        if len(roots) != 1:
            raise RuntimeError(f"Expected one SavedModel in {url}, found {len(roots)}")
        shutil.move(str(roots[0].parent), str(target))
    return target


class VisionModel:
    """Official exported model; input is RGB float32 NHWC in [0, 1]."""
    def __init__(self, family: str, variant: str, cache_dir="models", *, download=False):
        import tensorflow as tf
        self.name = checkpoint_name(family, variant)
        self.path = ensure_checkpoint(family, variant, cache_dir, download=download)
        self.model = tf.saved_model.load(str(self.path))
        self.forward = self.model.signatures["serving_default"]
        _, inputs = self.forward.structured_input_signature
        if "images" not in inputs:
            raise RuntimeError(f"Unexpected inputs for {self.name}: {inputs}")
        self.input_spec = inputs["images"]

    def predict(self, images):
        import numpy as np
        import tensorflow as tf
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0:
            raise ValueError("Expected a nonempty [batch, height, width, 3] RGB array")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Images must contain finite values in [0, 1]")
        expected = self.input_spec.shape.as_list()
        if any(want is not None and got != want for got, want in zip(images.shape, expected)):
            raise ValueError(f"Input shape {images.shape} does not match {expected}")
        return {key: value.numpy() for key, value in self.forward(
            images=tf.convert_to_tensor(images, dtype=self.input_spec.dtype)
        ).items()}

    def embed(self, images, representation="pre_logits"):
        outputs = self.predict(images)
        if representation not in outputs:
            raise KeyError(f"Missing {representation}; available outputs: {sorted(outputs)}")
        return outputs[representation]


def load_images(paths, size=(224, 224)):
    """RGB decode, direct bicubic resize, and scale to [0, 1]."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    images = []
    for path in paths:
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        images.append(cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0)
    if not images:
        raise ValueError("Supply at least one image")
    return np.stack(images)
