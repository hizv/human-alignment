"""Navon global-local similarity stimulus generation, embedding extraction, and analysis.

Stimulus rendering adapts `katherine-h/navon/generate_navon_stims.py`
(blob 98761af2680d16d39b50ed63e301ddbafb753c89) for Linux + Pillow 12 with
deterministic spatial shifts, upright default orientation, multiple local
size/spacing conditions, and isolated single-letter reference stimuli.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import string
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import FAMILIES, VARIANTS, checkpoint_name

IM_SIZE = 224
GLOBAL_FONT_SIZE = 160
DEFAULT_FONT_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
DEFAULT_LETTERS = list(string.ascii_uppercase)
CONDITIONS: dict[str, tuple[int, int]] = {
    "s10": (10, 11),
    "s14": (14, 15),
    "s18": (18, 19),
}
SHIFTS: list[tuple[int, int]] = [(0, 0), (16, 0), (-16, 0), (0, 16), (0, -16)]
PROVENANCE_NOTE = (
    "Adapted from katherine-h/navon/generate_navon_stims.py "
    "(blob 98761af2680d16d39b50ed63e301ddbafb753c89) using DejaVuSans-Bold.ttf, "
    "Pillow 12 font.getbbox centering, deterministic spatial shifts, 0-deg upright "
    "default, multiple local size/spacing conditions, and isolated-letter references."
)


def _draw_centered_letter(
    draw: ImageDraw.ImageDraw,
    letter: str,
    font: ImageFont.FreeTypeFont,
    center_x: float,
    center_y: float,
    fill: str = "black",
) -> None:
    left, top, right, bottom = font.getbbox(letter)
    w, h = right - left, bottom - top
    draw.text((center_x - w / 2 - left, center_y - h / 2 - top), letter, font=font, fill=fill)


def _measure_ink(image: Image.Image) -> tuple[float, list[int], bool]:
    arr = np.asarray(image)
    black = arr[:, :, 0] < 128
    ys, xs = np.where(black)
    if len(ys) == 0:
        return 0.0, [0, 0, 0, 0], False
    bbox = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
    im_h, im_w = arr.shape[:2]
    clipped = bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= im_h - 1 or bbox[3] >= im_w - 1
    return float(black.mean()), bbox, bool(clipped)


def letter_to_shifted_masks(
    letter: str,
    im_size: int = IM_SIZE,
    global_font_size: int = GLOBAL_FONT_SIZE,
    shifts: list[tuple[int, int]] = SHIFTS,
    font_path: str = DEFAULT_FONT_PATH,
) -> list[Image.Image]:
    """Render shifted global letter masks at deterministic positions."""
    font = ImageFont.truetype(font_path, global_font_size)
    masks: list[Image.Image] = []
    for shift_x, shift_y in shifts:
        image = Image.new("RGB", (im_size, im_size), "white")
        draw = ImageDraw.Draw(image)
        _draw_centered_letter(draw, letter, font, im_size / 2 + shift_x, im_size / 2 + shift_y)
        masks.append(image)
    return masks


def render_compound(
    mask_image: Image.Image,
    fill_letter: str,
    local_font_size: int,
    step_size: int | None = None,
    rotation_deg: float = 0.0,
    font_path: str = DEFAULT_FONT_PATH,
) -> tuple[Image.Image, dict]:
    """Tile local fill_letter inside dark pixels of mask_image."""
    mask = np.asarray(mask_image)
    im_h, im_w = mask.shape[:2]
    font = ImageFont.truetype(font_path, local_font_size)
    if step_size is None:
        left, top, right, bottom = font.getbbox(fill_letter)
        step_size = max(right - left, bottom - top) + 2

    image = Image.new("RGB", (im_w, im_h), "white")
    draw = ImageDraw.Draw(image)
    element_count = 0
    start = step_size // 2
    for y in range(start, im_h - start, step_size):
        for x in range(start, im_w - start, step_size):
            if mask[y, x, 0] < 128:
                _draw_centered_letter(draw, fill_letter, font, x, y)
                element_count += 1

    if rotation_deg != 0:
        image = image.rotate(rotation_deg, fillcolor="white", resample=Image.BILINEAR)

    ink_fraction, bbox, clipped = _measure_ink(image)
    return image, {
        "local_font_size": int(local_font_size),
        "step_size": int(step_size),
        "element_count": int(element_count),
        "ink_fraction": ink_fraction,
        "bbox": bbox,
        "clipped": clipped,
    }


def render_isolated(
    letter: str,
    scale: str,
    shift: tuple[int, int],
    im_size: int = IM_SIZE,
    global_font_size: int = GLOBAL_FONT_SIZE,
    local_font_size: int = 14,
    font_path: str = DEFAULT_FONT_PATH,
) -> tuple[Image.Image, dict]:
    """Render an isolated reference letter at global or local scale."""
    font_size = global_font_size if scale == "global" else local_font_size
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new("RGB", (im_size, im_size), "white")
    draw = ImageDraw.Draw(image)
    _draw_centered_letter(draw, letter, font, im_size / 2 + shift[0], im_size / 2 + shift[1])
    ink_fraction, bbox, clipped = _measure_ink(image)
    return image, {
        "local_font_size": int(font_size),
        "step_size": 0,
        "element_count": 1,
        "ink_fraction": ink_fraction,
        "bbox": bbox,
        "clipped": clipped,
    }


def _parse_letters(spec: str) -> list[str]:
    if spec.upper() == "ALL":
        return list(string.ascii_uppercase)
    letters = []
    for ch in spec.upper():
        if ch.isalpha() and ch not in letters:
            letters.append(ch)
    if len(letters) < 3:
        raise ValueError("Supply at least 3 distinct letters for Comparison B")
    return letters


def _make_contact_sheet(
    tiles: list[tuple[str, Image.Image]],
    cols: int,
    cell_size: int = IM_SIZE,
    label_height: int = 24,
) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, tile) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_size, r * (cell_size + label_height)
        draw.text((x0 + 4, y0 + 4), label, fill="black")
        sheet.paste(tile.resize((cell_size, cell_size)), (x0, y0 + label_height))
        draw.rectangle([x0, y0 + label_height, x0 + cell_size - 1, y0 + label_height + cell_size - 1], outline="#cccccc")
    return sheet


def _build_pilot_grids(out_dir: Path, items: list[dict], letters: list[str], num_positions: int) -> None:
    from .models import load_images

    g0 = "H" if "H" in letters else letters[0]
    l0 = "S" if "S" in letters and "S" != g0 else letters[1]
    by_key: dict[tuple, dict] = {}
    for item in items:
        if item["kind"] == "compound":
            by_key[("compound", item["condition"], item["pos_idx"], item["global_letter"], item["local_letter"])] = item
        elif item["ref_scale"] == "global":
            by_key[("iso_global", item["pos_idx"], item["global_letter"])] = item
        else:
            by_key[("iso_local", item["condition"], item["pos_idx"], item["local_letter"])] = item

    selected: list[tuple[str, Path]] = []
    for cond in CONDITIONS:
        for pos_idx in range(min(num_positions, 2)):
            for g, l in ((g0, l0), (l0, g0), (g0, g0)):
                rec = by_key[("compound", cond, pos_idx, g, l)]
                selected.append((f"{cond} p{pos_idx} C({g},{l})", out_dir / rec["rel_path"]))
    for pos_idx in range(min(num_positions, 2)):
        for letter in (g0, l0):
            rec = by_key[("iso_global", pos_idx, letter)]
            selected.append((f"global p{pos_idx} I({letter})", out_dir / rec["rel_path"]))
    for cond in CONDITIONS:
        for letter in (g0, l0):
            rec = by_key[("iso_local", cond, 0, letter)]
            selected.append((f"local {cond} p0 I({letter})", out_dir / rec["rel_path"]))

    orig_tiles = [(label, Image.open(path).convert("RGB")) for label, path in selected]
    cols = 6
    _make_contact_sheet(orig_tiles, cols=cols).save(out_dir / "pilot_grid_original.png")

    pre_arr = load_images([path for _, path in selected], size=(IM_SIZE, IM_SIZE))
    pre_tiles = [
        (label, Image.fromarray(np.clip(np.rint(pre_arr[i] * 255.0), 0, 255).astype(np.uint8)))
        for i, (label, _) in enumerate(selected)
    ]
    _make_contact_sheet(pre_tiles, cols=cols).save(out_dir / "pilot_grid_preprocessed.png")


def cmd_generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    comp_dir = out_dir / "images" / "compounds"
    iso_dir = out_dir / "images" / "isolated"
    comp_dir.mkdir(parents=True, exist_ok=True)
    iso_dir.mkdir(parents=True, exist_ok=True)

    letters = _parse_letters(args.letters)
    if not (1 <= args.num_positions <= len(SHIFTS)):
        raise ValueError(f"--num-positions must be in 1..{len(SHIFTS)}")
    shifts = SHIFTS[: args.num_positions]

    masks_by_letter = {
        letter: letter_to_shifted_masks(
            letter,
            im_size=IM_SIZE,
            global_font_size=GLOBAL_FONT_SIZE,
            shifts=shifts,
            font_path=args.font_path,
        )
        for letter in letters
    }

    items: list[dict] = []
    for pos_idx, shift in enumerate(shifts):
        for g in letters:
            img, meta = render_isolated(
                g, "global", shift, im_size=IM_SIZE, global_font_size=GLOBAL_FONT_SIZE, font_path=args.font_path
            )
            rel_path = f"images/isolated/global_p{pos_idx}_{g}.png"
            img.save(out_dir / rel_path)
            items.append({
                "image_id": f"iso_global_p{pos_idx}_{g}",
                "rel_path": rel_path,
                "kind": "isolated",
                "condition": "global",
                "pos_idx": pos_idx,
                "global_letter": g,
                "local_letter": "",
                "ref_scale": "global",
                **meta,
            })

    for cond, (local_font_size, step_size) in CONDITIONS.items():
        for pos_idx, shift in enumerate(shifts):
            for l in letters:
                img, meta = render_isolated(
                    l,
                    "local",
                    shift,
                    im_size=IM_SIZE,
                    global_font_size=GLOBAL_FONT_SIZE,
                    local_font_size=local_font_size,
                    font_path=args.font_path,
                )
                rel_path = f"images/isolated/local_{cond}_p{pos_idx}_{l}.png"
                img.save(out_dir / rel_path)
                items.append({
                    "image_id": f"iso_local_{cond}_p{pos_idx}_{l}",
                    "rel_path": rel_path,
                    "kind": "isolated",
                    "condition": cond,
                    "pos_idx": pos_idx,
                    "global_letter": "",
                    "local_letter": l,
                    "ref_scale": "local",
                    **meta,
                })

            for g in letters:
                mask_image = masks_by_letter[g][pos_idx]
                for l in letters:
                    img, meta = render_compound(
                        mask_image,
                        l,
                        local_font_size=local_font_size,
                        step_size=step_size,
                        rotation_deg=args.rotation_deg,
                        font_path=args.font_path,
                    )
                    rel_path = f"images/compounds/{cond}_p{pos_idx}_{g}_{l}.png"
                    img.save(out_dir / rel_path)
                    items.append({
                        "image_id": f"cmp_{cond}_p{pos_idx}_{g}_{l}",
                        "rel_path": rel_path,
                        "kind": "compound",
                        "condition": cond,
                        "pos_idx": pos_idx,
                        "global_letter": g,
                        "local_letter": l,
                        "ref_scale": "",
                        **meta,
                    })

    clipped_count = sum(1 for item in items if item["clipped"])
    zero_element_compounds = sum(1 for item in items if item["kind"] == "compound" and item["element_count"] == 0)
    manifest = {
        "config": {
            "letters": letters,
            "conditions": CONDITIONS,
            "shifts": shifts,
            "num_positions": args.num_positions,
            "im_size": IM_SIZE,
            "global_font_size": GLOBAL_FONT_SIZE,
            "font_path": args.font_path,
            "rotation_deg": args.rotation_deg,
        },
        "provenance": PROVENANCE_NOTE,
        "checks": {
            "total_images": len(items),
            "compound_images": sum(1 for item in items if item["kind"] == "compound"),
            "isolated_images": sum(1 for item in items if item["kind"] == "isolated"),
            "clipped_count": clipped_count,
            "zero_element_compounds": zero_element_compounds,
        },
        "items": items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _build_pilot_grids(out_dir, items, letters, args.num_positions)
    print(
        f"Generated {len(items)} images in {out_dir} "
        f"(clipped={clipped_count}, zero_elements={zero_element_compounds})",
        flush=True,
    )


def cmd_embed(args: argparse.Namespace) -> None:
    families = list(FAMILIES) if args.family == "all" else [args.family]
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if len(families) * len(variants) > 1:
        failed = []
        for family in families:
            for variant in variants:
                command = [
                    sys.executable,
                    "-m",
                    "human_alignment.navon",
                    "embed",
                    "--family",
                    family,
                    "--variant",
                    variant,
                    "--cache-dir",
                    args.cache_dir,
                    "--output-dir",
                    args.output_dir,
                    "--batch-size",
                    str(args.batch_size),
                ]
                if subprocess.run(command, check=False).returncode:
                    failed.append(f"{family}/{variant}")
        if failed:
            raise SystemExit(f"Failed: {', '.join(failed)}")
        return
    _embed_one(args)


def _embed_one(args: argparse.Namespace) -> None:
    import tensorflow as tf
    from .models import VisionModel, load_images

    out_dir = Path(args.output_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    model = VisionModel(args.family, args.variant, args.cache_dir, download=False)
    shape = model.input_spec.shape.as_list()
    height, width = shape[1] or IM_SIZE, shape[2] or IM_SIZE

    pre_chunks = []
    triplet_chunks = []
    paths = [out_dir / item["rel_path"] for item in items]
    for start in range(0, len(paths), args.batch_size):
        batch = load_images(paths[start : start + args.batch_size], size=(width, height))
        preds = model.predict(batch)
        for rep in ("pre_logits", "triplet_logits"):
            arr = preds[rep]
            if not np.isfinite(arr).all():
                raise RuntimeError(f"Non-finite {rep} output for {model.name}")
        pre_chunks.append(preds["pre_logits"].astype(np.float32))
        triplet_chunks.append(preds["triplet_logits"].astype(np.float32))

    pre_logits = np.concatenate(pre_chunks, axis=0)
    triplet_logits = np.concatenate(triplet_chunks, axis=0)

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    npz_path = emb_dir / f"{model.name}.npz"
    np.savez_compressed(
        npz_path,
        image_ids=np.array([item["image_id"] for item in items]),
        pre_logits=pre_logits,
        triplet_logits=triplet_logits,
    )
    meta = {
        "model": model.name,
        "family": args.family,
        "variant": args.variant,
        "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_logical_devices()],
        "n_images": len(items),
        "pre_logits_shape": list(pre_logits.shape),
        "triplet_logits_shape": list(triplet_logits.shape),
    }
    (emb_dir / f"{model.name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"EMBED {model.name}: {npz_path} ({pre_logits.shape})", flush=True)


def _load_embeddings(emb_dir: Path, manifest_ids: list[str]) -> dict[tuple[str, str], dict]:
    name_to_fv = {checkpoint_name(f, v): (f, v) for f in FAMILIES for v in VARIANTS}
    loaded: dict[tuple[str, str], dict] = {}
    for npz_path in sorted(emb_dir.glob("*.npz")):
        model_name = npz_path.stem
        if model_name not in name_to_fv:
            continue
        data = np.load(npz_path, allow_pickle=False)
        saved_ids = data["image_ids"].tolist()
        if saved_ids != manifest_ids:
            raise RuntimeError(f"Image ID order mismatch in {npz_path}")
        family, variant = name_to_fv[model_name]
        loaded[(family, variant)] = {
            "model_name": model_name,
            "pre_logits": data["pre_logits"],
            "triplet_logits": data["triplet_logits"],
        }
    if not loaded:
        raise FileNotFoundError(f"No checkpoint embeddings found in {emb_dir}; run `alignment-navon embed` first.")
    return loaded


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def _choice(margin: float) -> str:
    if margin > 0:
        return "global"
    if margin < 0:
        return "local"
    return "tie"


def _summarize_group(trials: list[dict]) -> dict:
    margins = np.array([t["margin"] for t in trials], dtype=np.float64)
    by_pair: dict[str, list[float]] = {}
    for t in trials:
        by_pair.setdefault(t["pair_id"], []).append(t["margin"])
    pair_means = [float(np.mean(vals)) for vals in by_pair.values()]
    return {
        "n_trials": len(trials),
        "n_pairs": len(by_pair),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "pair_balanced_mean_margin": float(np.mean(pair_means)),
        "global_frac": float(np.mean(margins > 0)),
        "local_frac": float(np.mean(margins < 0)),
        "tie_frac": float(np.mean(margins == 0)),
    }


def _paired_contrast(
    trials_a: list[dict],
    trials_b: list[dict],
    letters: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    map_a = {t["trial_key"]: t for t in trials_a}
    map_b = {t["trial_key"]: t for t in trials_b}
    keys = sorted(set(map_a) & set(map_b))
    if len(keys) != len(trials_a) or len(keys) != len(trials_b):
        raise RuntimeError("Mismatched trial keys between variants")

    diff_margin = np.array([map_a[k]["margin"] - map_b[k]["margin"] for k in keys], dtype=np.float64)
    diff_choice = np.array(
        [float(map_a[k]["margin"] > 0) - float(map_b[k]["margin"] > 0) for k in keys], dtype=np.float64
    )
    pair_ids = [map_a[k]["pair_id"] for k in keys]
    involved = [map_a[k]["involved_letters"] for k in keys]

    cluster_idx: dict[str, list[int]] = {}
    for idx, pid in enumerate(pair_ids):
        cluster_idx.setdefault(pid, []).append(idx)
    clusters = list(cluster_idx.values())
    c_margin = np.array([diff_margin[idxs].mean() for idxs in clusters], dtype=np.float64)
    c_choice = np.array([diff_choice[idxs].mean() for idxs in clusters], dtype=np.float64)
    c_weights = np.array([len(idxs) for idxs in clusters], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    boot_margin = np.empty(n_boot, dtype=np.float64)
    boot_choice = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        w = c_weights[draw]
        boot_margin[b] = np.average(c_margin[draw], weights=w)
        boot_choice[b] = np.average(c_choice[draw], weights=w)

    lo_m, hi_m = np.percentile(boot_margin, [2.5, 97.5])
    lo_c, hi_c = np.percentile(boot_choice, [2.5, 97.5])

    jack_m = []
    jack_c = []
    for letter in letters:
        keep = [i for i, inv in enumerate(involved) if letter not in inv]
        if keep:
            jack_m.append(float(diff_margin[keep].mean()))
            jack_c.append(float(diff_choice[keep].mean()))
    k_eff = len(jack_m)
    jack_se_m = (
        float(np.sqrt((k_eff - 1) / k_eff * np.sum((np.array(jack_m) - np.mean(jack_m)) ** 2)))
        if k_eff > 1
        else 0.0
    )
    jack_se_c = (
        float(np.sqrt((k_eff - 1) / k_eff * np.sum((np.array(jack_c) - np.mean(jack_c)) ** 2)))
        if k_eff > 1
        else 0.0
    )

    return {
        "n_trials": len(keys),
        "n_clusters": n_clusters,
        "delta_margin": float(diff_margin.mean()),
        "margin_ci95_low": float(lo_m),
        "margin_ci95_high": float(hi_m),
        "margin_jackknife_se": jack_se_m,
        "delta_global_frac": float(diff_choice.mean()),
        "global_frac_ci95_low": float(lo_c),
        "global_frac_ci95_high": float(hi_c),
        "global_frac_jackknife_se": jack_se_c,
    }


def _render_summary_figure(summary_rows: list[dict], fig_path: Path, representation: str = "pre_logits") -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    subconditions = sorted(
        {(r["comparison"], r["subcondition"]) for r in summary_rows if r["representation"] == representation}
    )
    if not subconditions:
        return

    lookup = {
        (r["family"], r["variant"], r["comparison"], r["subcondition"]): r
        for r in summary_rows
        if r["representation"] == representation
    }
    colors = {"base": "#6c757d", "unalignet": "#d97706", "alignet": "#2563eb"}
    families = [f for f in ("dino", "siglip") if any(r["family"] == f for r in summary_rows)]
    if not families:
        return

    panel_w = max(560, 120 + len(subconditions) * 95)
    panel_h = 340
    img = Image.new("RGB", (panel_w, panel_h * len(families) + 60), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 14), f"Navon mean similarity margin ({representation})", fill="black")

    lx = panel_w - 260
    for idx, v in enumerate(("base", "unalignet", "alignet")):
        draw.rectangle([lx + idx * 82, 14, lx + idx * 82 + 12, 26], fill=colors[v])
        draw.text((lx + idx * 82 + 16, 14), v, fill="black")

    all_vals = [
        r["mean_margin"] for r in summary_rows if r["representation"] == representation
    ]
    max_abs = max([0.05] + [abs(v) for v in all_vals]) * 1.15

    for f_idx, family in enumerate(families):
        top = 50 + f_idx * panel_h
        plot_l, plot_r = 75, panel_w - 30
        plot_t, plot_b = top + 35, top + panel_h - 65
        zero_y = int((plot_t + plot_b) / 2)
        draw.text((20, top + 8), f"{FAMILIES[family]} ({family})", fill="black")
        draw.line([plot_l, plot_t, plot_l, plot_b], fill="#444444", width=1)
        draw.line([plot_l, zero_y, plot_r, zero_y], fill="#888888", width=1)
        draw.text((12, plot_t - 4), f"+{max_abs:.2f}", fill="#444444")
        draw.text((12, zero_y - 6), " 0.00", fill="#444444")
        draw.text((12, plot_b - 6), f"-{max_abs:.2f}", fill="#444444")

        group_w = (plot_r - plot_l) / len(subconditions)
        bar_w = max(8, int(group_w * 0.22))
        for s_idx, (comp, sub) in enumerate(subconditions):
            gx = plot_l + (s_idx + 0.5) * group_w
            short_comp = "A" if comp == "comparison_a" else "B"
            draw.text((int(gx - group_w * 0.4), plot_b + 8), f"{short_comp}:{sub}", fill="#222222")
            for v_idx, variant in enumerate(("base", "unalignet", "alignet")):
                row = lookup.get((family, variant, comp, sub))
                if row is None:
                    continue
                val = row["mean_margin"]
                y_val = int(zero_y - (val / max_abs) * ((plot_b - plot_t) / 2))
                bx0 = int(gx + (v_idx - 1.5) * (bar_w + 2))
                bx1 = bx0 + bar_w
                y0, y1 = min(zero_y, y_val), max(zero_y, y_val)
                draw.rectangle([bx0, y0, bx1, max(y0 + 1, y1)], fill=colors[variant])

    img.save(fig_path)


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    manifest_ids = [item["image_id"] for item in items]
    letters = manifest["config"]["letters"]
    conditions = list(manifest["config"]["conditions"])
    positions = list(range(manifest["config"]["num_positions"]))

    embeddings = _load_embeddings(out_dir / "embeddings", manifest_ids)

    compound_idx: dict[tuple[str, int, str, str], int] = {}
    iso_global_idx: dict[tuple[int, str], int] = {}
    iso_local_idx: dict[tuple[str, int, str], int] = {}
    for idx, item in enumerate(items):
        if item["kind"] == "compound":
            compound_idx[(item["condition"], item["pos_idx"], item["global_letter"], item["local_letter"])] = idx
        elif item["ref_scale"] == "global":
            iso_global_idx[(item["pos_idx"], item["global_letter"])] = idx
        else:
            iso_local_idx[(item["condition"], item["pos_idx"], item["local_letter"])] = idx

    trials_a: list[dict] = []
    trials_b: list[dict] = []
    grouped_trials: dict[tuple[str, str, str, str, str], list[dict]] = {}

    b_regimes = [("same_pos", 0, 0)]
    if len(positions) >= 2:
        b_regimes.append(("cross_pos", 0, 1))

    for (family, variant), emb_bundle in sorted(embeddings.items()):
        for rep in args.representations:
            unit = _unit_rows(emb_bundle[rep])

            # Comparison A: compound to isolated letters
            for cond in conditions:
                for pos_idx in positions:
                    for ref_scale in ("global", "local"):
                        sub = f"{cond}|{ref_scale}"
                        for g in letters:
                            for l in letters:
                                if g == l:
                                    continue
                                c_idx = compound_idx[(cond, pos_idx, g, l)]
                                if ref_scale == "global":
                                    rg_idx = iso_global_idx[(pos_idx, g)]
                                    rl_idx = iso_global_idx[(pos_idx, l)]
                                else:
                                    rg_idx = iso_local_idx[(cond, pos_idx, g)]
                                    rl_idx = iso_local_idx[(cond, pos_idx, l)]
                                sim_g = float(unit[c_idx] @ unit[rg_idx])
                                sim_l = float(unit[c_idx] @ unit[rl_idx])
                                margin = sim_g - sim_l
                                pid = "_".join(sorted([g, l]))
                                row = {
                                    "family": family,
                                    "variant": variant,
                                    "representation": rep,
                                    "condition": cond,
                                    "pos_idx": pos_idx,
                                    "ref_scale": ref_scale,
                                    "global_letter": g,
                                    "local_letter": l,
                                    "pair_id": pid,
                                    "involved_letters": f"{g}{l}",
                                    "trial_key": f"{cond}|p{pos_idx}|{ref_scale}|{g}|{l}",
                                    "sim_global": sim_g,
                                    "sim_local": sim_l,
                                    "margin": margin,
                                    "choice": _choice(margin),
                                }
                                trials_a.append(row)
                                grouped_trials.setdefault((family, variant, rep, "comparison_a", sub), []).append(row)
                                grouped_trials.setdefault(
                                    (family, variant, rep, "comparison_a", f"ALL|{ref_scale}"), []
                                ).append(row)

            # Comparison B: compound to matching compounds
            for cond in conditions:
                for regime, pos_ref, pos_cand in b_regimes:
                    sub = f"{cond}|{regime}"
                    for g in letters:
                        for l in letters:
                            if g == l:
                                continue
                            ref_idx = compound_idx[(cond, pos_ref, g, l)]
                            pid = "_".join(sorted([g, l]))
                            for t in letters:
                                if t in (g, l):
                                    continue
                                cg_idx = compound_idx[(cond, pos_cand, g, t)]
                                cl_idx = compound_idx[(cond, pos_cand, t, l)]
                                sim_g = float(unit[ref_idx] @ unit[cg_idx])
                                sim_l = float(unit[ref_idx] @ unit[cl_idx])
                                margin = sim_g - sim_l
                                row = {
                                    "family": family,
                                    "variant": variant,
                                    "representation": rep,
                                    "condition": cond,
                                    "regime": regime,
                                    "pos_ref": pos_ref,
                                    "pos_cand": pos_cand,
                                    "global_letter": g,
                                    "local_letter": l,
                                    "third_letter": t,
                                    "pair_id": pid,
                                    "involved_letters": f"{g}{l}{t}",
                                    "trial_key": f"{cond}|{regime}|{g}|{l}|{t}",
                                    "sim_global": sim_g,
                                    "sim_local": sim_l,
                                    "margin": margin,
                                    "choice": _choice(margin),
                                }
                                trials_b.append(row)
                                grouped_trials.setdefault((family, variant, rep, "comparison_b", sub), []).append(row)
                                grouped_trials.setdefault(
                                    (family, variant, rep, "comparison_b", f"ALL|{regime}"), []
                                ).append(row)

    for fname, rows in (("trials_comparison_a.csv", trials_a), ("trials_comparison_b.csv", trials_b)):
        if rows:
            with (out_dir / fname).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    summary_rows: list[dict] = []
    for (family, variant, rep, comp, sub), group in sorted(grouped_trials.items()):
        summary_rows.append({
            "family": family,
            "variant": variant,
            "representation": rep,
            "comparison": comp,
            "subcondition": sub,
            **_summarize_group(group),
        })

    if summary_rows:
        with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    contrast_rows: list[dict] = []
    slice_keys = sorted({(rep, comp, sub) for (_, _, rep, comp, sub) in grouped_trials})
    for family in FAMILIES:
        for rep, comp, sub in slice_keys:
            for va, vb in (("alignet", "unalignet"), ("alignet", "base"), ("unalignet", "base")):
                ga = grouped_trials.get((family, va, rep, comp, sub))
                gb = grouped_trials.get((family, vb, rep, comp, sub))
                if not ga or not gb:
                    continue
                stats = _paired_contrast(ga, gb, letters, n_boot=args.n_boot, seed=args.seed)
                contrast_rows.append({
                    "family": family,
                    "contrast": f"{va}_minus_{vb}",
                    "representation": rep,
                    "comparison": comp,
                    "subcondition": sub,
                    **stats,
                })

    report = {
        "representations": args.representations,
        "primary_representation": "pre_logits",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "summary": summary_rows,
        "paired_contrasts": contrast_rows,
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    primary_rep = "pre_logits" if "pre_logits" in args.representations else args.representations[0]
    _render_summary_figure(summary_rows, out_dir / "figures" / "margin_summary.png", representation=primary_rep)
    _write_run_record(out_dir, manifest, embeddings, summary_rows, contrast_rows)
    print(f"Analyzed {len(trials_a)} Comp-A and {len(trials_b)} Comp-B trials per representation -> {out_dir}", flush=True)


def _write_run_record(
    out_dir: Path,
    manifest: dict,
    embeddings: dict,
    summary_rows: list[dict],
    contrast_rows: list[dict],
) -> None:
    cfg = manifest["config"]
    chk = manifest["checks"]
    lines = [
        "# Navon Global-Local Experiment Run Record",
        "",
        "## Provenance and Configuration",
        f"- **Stimulus generator**: {manifest['provenance']}",
        f"- **Letters ({len(cfg['letters'])})**: `{' '.join(cfg['letters'])}`",
        f"- **Conditions (font_size, step_size)**: `{cfg['conditions']}`",
        f"- **Positions**: `{cfg['num_positions']}` deterministic shifts `{cfg['shifts']}` (rotation `{cfg['rotation_deg']}` deg)",
        f"- **Images**: `{chk['total_images']}` total (`{chk['compound_images']}` compounds, `{chk['isolated_images']}` isolated; clipped=`{chk['clipped_count']}`, zero_elements=`{chk['zero_element_compounds']}`)",
        "- **Preprocessing**: `224x224` RGB bicubic, float32 scaled to `[0, 1]` via `human_alignment.models.load_images` (note: public AligNet docs do not specify a separate DINOv2-B inference preprocessing recipe, so both families share this convention).",
        "- **Representation**: `pre_logits` (768-d) is primary across `base`, `unalignet`, and `alignet`; `triplet_logits` (1024-d) is secondary because it comes from the alignment projection head (untrained in `base`).",
        f"- **Checkpoints evaluated**: {', '.join(bundle['model_name'] for bundle in embeddings.values())}",
        "",
        "## Primary Paired Contrasts (`pre_logits`, aggregate `ALL|*` slices)",
        "",
        "| Family | Contrast | Comparison | Slice | Delta Margin | 95% Bootstrap CI | Jackknife SE | Delta Global Frac |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in contrast_rows:
        if row["representation"] != "pre_logits" or not row["subcondition"].startswith("ALL|"):
            continue
        lines.append(
            f"| {row['family']} | {row['contrast']} | {row['comparison']} | {row['subcondition']} | "
            f"{row['delta_margin']:+.4f} | [{row['margin_ci95_low']:+.4f}, {row['margin_ci95_high']:+.4f}] | "
            f"{row['margin_jackknife_se']:.4f} | {row['delta_global_frac']:+.4f} |"
        )
    lines.extend([
        "",
        "## Generated Artifacts",
        "- `manifest.json`, `pilot_grid_original.png`, `pilot_grid_preprocessed.png`",
        "- `embeddings/*.npz` and `embeddings/*.json`",
        "- `trials_comparison_a.csv`, `trials_comparison_b.csv`",
        "- `summary_metrics.csv`, `summary_metrics.json`, `figures/margin_summary.png`",
        "",
    ])
    (out_dir / "run_record.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p_gen = subparsers.add_parser("generate", help="Generate Navon stimuli, manifest, and pilot contact sheets")
    p_gen.add_argument("--output-dir", default="outputs/navon")
    p_gen.add_argument("--letters", default="".join(DEFAULT_LETTERS), help="Letters string or ALL for A-Z")
    p_gen.add_argument("--num-positions", type=int, default=2)
    p_gen.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    p_gen.add_argument("--rotation-deg", type=float, default=0.0)
    p_gen.set_defaults(func=cmd_generate)

    p_emb = subparsers.add_parser("embed", help="Extract embeddings for generated Navon stimuli")
    p_emb.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    p_emb.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    p_emb.add_argument("--cache-dir", default="models")
    p_emb.add_argument("--output-dir", default="outputs/navon")
    p_emb.add_argument("--batch-size", type=int, default=64)
    p_emb.set_defaults(func=cmd_embed)

    p_ana = subparsers.add_parser("analyze", help="Score Comparison A & B and compute paired contrasts")
    p_ana.add_argument("--output-dir", default="outputs/navon")
    p_ana.add_argument("--representations", nargs="+", default=["pre_logits", "triplet_logits"])
    p_ana.add_argument("--n-boot", type=int, default=1000)
    p_ana.add_argument("--seed", type=int, default=0)
    p_ana.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
