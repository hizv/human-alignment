"""Visual-first exploratory analysis and plotting helpers for the Navon experiment.

All charts are rendered with Pillow + DejaVu Sans TrueType fonts so the module
runs directly inside the project's `.venv` without extra plotting packages.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import FAMILIES, VARIANTS, checkpoint_name, load_images

FONT_REG_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"

VARIANT_ORDER = ("base", "unalignet", "alignet")
VARIANT_LABELS = {"base": "Base", "unalignet": "UnAligNet", "alignet": "AligNet"}
VARIANT_COLORS = {
    "base": (100, 116, 139),       # slate #64748b
    "unalignet": (217, 119, 6),    # amber #d97706
    "alignet": (37, 99, 235),      # blue  #2563eb
}
TEXT_DARK = (24, 24, 27)
TEXT_MUTED = (82, 82, 91)
GRID_LINE = (228, 228, 231)
AXIS_LINE = (113, 113, 122)
BG_LIGHT = (248, 250, 252)


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_REG_PATH, size)


def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return int(right - left), int(bottom - top)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = TEXT_DARK,
    anchor: str = "la",
) -> None:
    w, h = _text_size(text, font)
    left, top, _, _ = font.getbbox(text)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h / 2
    elif anchor[1] == "b":
        y -= h
    draw.text((x - left, y - top), text, font=font, fill=fill)


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 1,
    dash: int = 6,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1:
        return
    steps = max(1, int(dist // (dash * 2)))
    for i in range(steps + 1):
        t0 = min(1.0, (i * 2 * dash) / dist)
        t1 = min(1.0, ((i * 2 + 1) * dash) / dist)
        if t1 > t0:
            draw.line(
                [x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
                fill=fill,
                width=width,
            )


def _resolve_out_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    cwd = Path.cwd()
    for candidate in (cwd / "outputs" / "navon", cwd.parent / "outputs" / "navon"):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate outputs/navon/manifest.json from current directory.")


@lru_cache(maxsize=4)
def load_workspace(out_dir_str: str | None = None) -> dict:
    """Load manifest, summary metrics, and normalized checkpoint embeddings once."""
    out_dir = _resolve_out_dir(out_dir_str)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    manifest_ids = [item["image_id"] for item in items]

    letters = manifest["config"]["letters"]
    letter_to_i = {ch: i for i, ch in enumerate(letters)}
    conditions = list(manifest["config"]["conditions"].keys())
    num_positions = int(manifest["config"]["num_positions"])

    comp_idx: dict[tuple[str, int, int, int], int] = {}
    iso_g_idx: dict[tuple[int, int], int] = {}
    iso_l_idx: dict[tuple[str, int, int], int] = {}
    item_by_id: dict[str, dict] = {}
    for idx, item in enumerate(items):
        item_by_id[item["image_id"]] = item
        if item["kind"] == "compound":
            comp_idx[(item["condition"], item["pos_idx"], letter_to_i[item["global_letter"]], letter_to_i[item["local_letter"]])] = idx
        elif item["ref_scale"] == "global":
            iso_g_idx[(item["pos_idx"], letter_to_i[item["global_letter"]])] = idx
        else:
            iso_l_idx[(item["condition"], item["pos_idx"], letter_to_i[item["local_letter"]])] = idx

    emb_dir = out_dir / "embeddings"
    unit: dict[tuple[str, str, str], np.ndarray] = {}
    for family in FAMILIES:
        for variant in VARIANTS:
            npz_path = emb_dir / f"{checkpoint_name(family, variant)}.npz"
            if not npz_path.is_file():
                continue
            data = np.load(npz_path, allow_pickle=False)
            if data["image_ids"].tolist() != manifest_ids:
                raise RuntimeError(f"Image ID order mismatch in {npz_path}")
            for rep in summary["representations"]:
                vecs = data[rep].astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                unit[(family, variant, rep)] = vecs / np.maximum(norms, 1e-12)

    pairs = [(i, j) for i in range(len(letters)) for j in range(i + 1, len(letters))]
    summary_lookup = {
        (r["family"], r["variant"], r["representation"], r["comparison"], r["subcondition"]): r
        for r in summary["summary"]
    }
    contrast_lookup = {
        (r["family"], r["contrast"], r["representation"], r["comparison"], r["subcondition"]): r
        for r in summary["paired_contrasts"]
    }

    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "summary": summary,
        "summary_lookup": summary_lookup,
        "contrast_lookup": contrast_lookup,
        "items": items,
        "item_by_id": item_by_id,
        "letters": letters,
        "letter_to_i": letter_to_i,
        "conditions": conditions,
        "num_positions": num_positions,
        "comp_idx": comp_idx,
        "iso_g_idx": iso_g_idx,
        "iso_l_idx": iso_l_idx,
        "unit": unit,
        "pairs": pairs,
    }


def _load_preprocessed_tile(store: dict, rel_path: str, size: int = 190) -> Image.Image:
    arr = load_images([store["out_dir"] / rel_path], size=(224, 224))[0]
    img = Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8))
    return img.resize((size, size), Image.Resampling.NEAREST) if size != 224 else img


def _comp_a_pair_matrix(
    store: dict,
    family: str,
    variant: str,
    rep: str,
    conds: list[str] | None = None,
    ref_scales: tuple[str, ...] = ("global", "local"),
    positions: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-slice arrays and 26x26 ordered-pair margin matrix for Comparison A."""
    u = store["unit"][(family, variant, rep)]
    k = len(store["letters"])
    conds = conds or store["conditions"]
    positions = positions if positions is not None else list(range(store["num_positions"]))

    sum_mat = np.zeros((k, k), dtype=np.float64)
    cnt_mat = np.zeros((k, k), dtype=np.int64)
    all_margins: list[np.ndarray] = []
    all_pair_ids: list[np.ndarray] = []

    pair_lookup = np.full((k, k), -1, dtype=np.int64)
    for pid, (i, j) in enumerate(store["pairs"]):
        pair_lookup[i, j] = pid
        pair_lookup[j, i] = pid

    off_g, off_l = np.where(~np.eye(k, dtype=bool))
    pids_off = pair_lookup[off_g, off_l]

    for cond in conds:
        for pos in positions:
            c_ids = np.array([[store["comp_idx"][(cond, pos, g, l)] for l in range(k)] for g in range(k)])
            c_vecs = u[c_ids]  # (K, K, D)
            for scale in ref_scales:
                if scale == "global":
                    r_ids = np.array([store["iso_g_idx"][(pos, x)] for x in range(k)])
                else:
                    r_ids = np.array([store["iso_l_idx"][(cond, pos, x)] for x in range(k)])
                r_vecs = u[r_ids]  # (K, D)
                sim_g = np.einsum("gld,gd->gl", c_vecs, r_vecs)
                sim_l = np.einsum("gld,ld->gl", c_vecs, r_vecs)
                margin = sim_g - sim_l
                sum_mat[off_g, off_l] += margin[off_g, off_l]
                cnt_mat[off_g, off_l] += 1
                all_margins.append(margin[off_g, off_l])
                all_pair_ids.append(pids_off)

    mean_mat = np.divide(sum_mat, np.maximum(cnt_mat, 1), out=np.full((k, k), np.nan), where=cnt_mat > 0)
    return np.concatenate(all_margins), np.concatenate(all_pair_ids), mean_mat


def _comp_b_pair_matrix(
    store: dict,
    family: str,
    variant: str,
    rep: str,
    conds: list[str] | None = None,
    regimes: tuple[str, ...] = ("same_pos", "cross_pos"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized trial margins, pair IDs, and 26x26 mean margin matrix for Comparison B."""
    u = store["unit"][(family, variant, rep)]
    k = len(store["letters"])
    conds = conds or store["conditions"]

    regime_pos = {"same_pos": (0, 0)}
    if store["num_positions"] >= 2:
        regime_pos["cross_pos"] = (0, 1)

    pair_lookup = np.full((k, k), -1, dtype=np.int64)
    for pid, (i, j) in enumerate(store["pairs"]):
        pair_lookup[i, j] = pid
        pair_lookup[j, i] = pid

    gg, ll, tt = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")
    valid = (gg != ll) & (tt != gg) & (tt != ll)
    vg, vl, vt = gg[valid], ll[valid], tt[valid]
    vpids = pair_lookup[vg, vl]

    sum_mat = np.zeros((k, k), dtype=np.float64)
    cnt_mat = np.zeros((k, k), dtype=np.int64)
    all_margins: list[np.ndarray] = []
    all_pair_ids: list[np.ndarray] = []

    for cond in conds:
        for reg in regimes:
            if reg not in regime_pos:
                continue
            pos_ref, pos_cand = regime_pos[reg]
            ref_ids = np.array([store["comp_idx"][(cond, pos_ref, g, l)] for g in range(k) for l in range(k)])
            cand_ids = np.array([store["comp_idx"][(cond, pos_cand, g, l)] for g in range(k) for l in range(k)])
            m4 = (u[ref_ids] @ u[cand_ids].T).reshape(k, k, k, k)
            margin_3d = m4[vg, vl, vt]  # wait: sim_g is m4[g, l, g, t], sim_l is m4[g, l, t, l]
            sim_g = m4[vg, vl, vg, vt]
            sim_l = m4[vg, vl, vt, vl]
            margins = sim_g - sim_l
            np.add.at(sum_mat, (vg, vl), margins)
            np.add.at(cnt_mat, (vg, vl), 1)
            all_margins.append(margins)
            all_pair_ids.append(vpids)

    mean_mat = np.divide(sum_mat, np.maximum(cnt_mat, 1), out=np.full((k, k), np.nan), where=cnt_mat > 0)
    return np.concatenate(all_margins), np.concatenate(all_pair_ids), mean_mat


def _unordered_pair_means(margins: np.ndarray, pair_ids: np.ndarray, n_pairs: int = 325) -> np.ndarray:
    sums = np.bincount(pair_ids, weights=margins, minlength=n_pairs)
    cnts = np.bincount(pair_ids, minlength=n_pairs)
    return sums / np.maximum(cnts, 1)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, variants=VARIANT_ORDER) -> None:
    cx = x
    f = _font(14, bold=True)
    for v in variants:
        draw.ellipse([cx, y + 2, cx + 12, y + 14], fill=VARIANT_COLORS[v])
        label = VARIANT_LABELS[v]
        _draw_text(draw, (cx + 18, y), label, f, fill=TEXT_DARK)
        cx += 26 + _text_size(label, f)[0] + 16


# ---------------------------------------------------------------------------
# 1. What images are we comparing?
# ---------------------------------------------------------------------------
def show_stimulus_gallery(
    global_letter: str = "H",
    local_letter: str = "S",
    third_letter: str = "T",
    condition: str = "s14",
    pos_idx: int = 0,
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Display preprocessed 224x224 inputs used in Comparison A and Comparison B."""
    store = load_workspace(str(out_dir) if out_dir else None)
    g, l, t = global_letter.upper(), local_letter.upper(), third_letter.upper()
    if len({g, l, t}) < 3:
        raise ValueError("Choose three distinct letters for global_letter, local_letter, and third_letter.")

    item_by_id = store["item_by_id"]
    shift_pos1 = 1 if store["num_positions"] >= 2 else pos_idx

    row1_ids = [
        ("Target compound C(G, L)", f"cmp_{condition}_p{pos_idx}_{g}_{l}"),
        ("Reversed compound C(L, G)", f"cmp_{condition}_p{pos_idx}_{l}_{g}"),
        (f"Isolated global I({g})", f"iso_global_p{pos_idx}_{g}"),
        (f"Isolated global I({l})", f"iso_global_p{pos_idx}_{l}"),
        (f"Isolated local I({g})", f"iso_local_{condition}_p{pos_idx}_{g}"),
        (f"Isolated local I({l})", f"iso_local_{condition}_p{pos_idx}_{l}"),
    ]
    row2_ids = [
        ("Reference C(G, L) [p0]", f"cmp_{condition}_p0_{g}_{l}"),
        ("Global match C(G, T) [p0]", f"cmp_{condition}_p0_{g}_{t}"),
        ("Local match C(T, L) [p0]", f"cmp_{condition}_p0_{t}_{l}"),
        (f"Global match C(G, T) [p{shift_pos1}]", f"cmp_{condition}_p{shift_pos1}_{g}_{t}"),
        (f"Local match C(T, L) [p{shift_pos1}]", f"cmp_{condition}_p{shift_pos1}_{t}_{l}"),
    ]

    tile_s = 190
    pad_x, gap_x = 28, 18
    cols = 6
    width = pad_x * 2 + cols * tile_s + (cols - 1) * gap_x
    height = 720
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    cfg = store["manifest"]["config"]
    _draw_text(
        draw,
        (pad_x, 18),
        f"1. Preprocessed 224x224 Model Inputs  (G={g}, L={l}, T={t} | cond={condition} | rot={cfg['rotation_deg']}°)",
        _font(20, bold=True),
    )
    _draw_text(
        draw,
        (pad_x, 46),
        "Images shown are reconstructed directly from load_images(..., size=(224, 224)) [RGB bicubic float32 in [0, 1]].",
        _font(13),
        fill=TEXT_MUTED,
    )

    def draw_row(title: str, y_top: int, specs: list[tuple[str, str]]) -> None:
        _draw_text(draw, (pad_x, y_top), title, _font(15, bold=True), fill=TEXT_DARK)
        y_img = y_top + 42
        for idx, (caption, img_id) in enumerate(specs):
            rec = item_by_id[img_id]
            x0 = pad_x + idx * (tile_s + gap_x)
            _draw_text(draw, (x0 + tile_s / 2, y_img - 6), caption, _font(12, bold=True), anchor="mb")
            tile = _load_preprocessed_tile(store, rec["rel_path"], size=tile_s)
            canvas.paste(tile, (x0, y_img))
            draw.rectangle([x0, y_img, x0 + tile_s - 1, y_img + tile_s - 1], outline=(180, 180, 188), width=1)
            shift = cfg["shifts"][rec["pos_idx"]]
            meta1 = f"size={rec['local_font_size']}px step={rec['step_size']} pos=({shift[0]},{shift[1]})"
            meta2 = f"elems={rec['element_count']} ink={rec['ink_fraction']*100:.1f}%"
            _draw_text(draw, (x0 + tile_s / 2, y_img + tile_s + 6), meta1, _font(11), fill=TEXT_MUTED, anchor="ma")
            _draw_text(draw, (x0 + tile_s / 2, y_img + tile_s + 21), meta2, _font(11), fill=TEXT_MUTED, anchor="ma")

    draw_row(f"Comparison A — Compound vs. Isolated References (pos_idx={pos_idx})", 82, row1_ids)
    draw_row("Comparison B — Compound Reference vs. Matching Compounds (same_pos p0 vs. cross_pos p1)", 398, row2_ids)

    # Info card in slot 6 of row 2
    ix0 = pad_x + 5 * (tile_s + gap_x)
    iy0 = 398 + 42
    draw.rectangle([ix0, iy0, ix0 + tile_s - 1, iy0 + tile_s - 1], fill=BG_LIGHT, outline=(203, 213, 225))
    info_lines = [
        "Margin definition:",
        "margin =",
        "  cos(ref, global)",
        "  - cos(ref, local)",
        "",
        "> 0 : closer to global",
        "< 0 : closer to local",
        "= 0 : tie",
        "",
        "p0 = (0, 0) center",
        "p1 = (+16, 0) shift",
    ]
    for li, text in enumerate(info_lines):
        _draw_text(draw, (ix0 + 12, iy0 + 12 + li * 15), text, _font(12, bold=(li == 0)), fill=TEXT_DARK)

    return canvas


# ---------------------------------------------------------------------------
# 2. How does one example behave?
# ---------------------------------------------------------------------------
def select_example_by_rule(
    rule: str = "predetermined",
    representation: str = "pre_logits",
    family: str = "dino",
    variant: str = "alignet",
    condition: str = "s14",
    out_dir: str | Path | None = None,
) -> tuple[str, str, str, str]:
    """Select an example (G, L, T) using a transparent, explicitly labeled rule."""
    if rule == "predetermined":
        return "H", "S", "T", "Predetermined example: C(H, S) with T=T"

    store = load_workspace(str(out_dir) if out_dir else None)
    letters = store["letters"]
    k = len(letters)
    _, _, mat_var = _comp_a_pair_matrix(store, family, variant, representation, conds=[condition], ref_scales=("global",), positions=[0])
    _, _, mat_un = _comp_a_pair_matrix(store, family, "unalignet", representation, conds=[condition], ref_scales=("global",), positions=[0])
    diff_mat = mat_var - mat_un

    mask = ~np.eye(k, dtype=bool)
    gs, ls = np.where(mask)
    vals = mat_var[gs, ls]
    diffs = diff_mat[gs, ls]

    if rule == "most_global":
        idx = int(np.argmax(vals))
        desc = f"Extreme high (most global Comp-A {condition} global-ref for {family}/{variant}: {vals[idx]:+.4f})"
    elif rule == "most_local":
        idx = int(np.argmin(vals))
        desc = f"Extreme low (most local Comp-A {condition} global-ref for {family}/{variant}: {vals[idx]:+.4f})"
    elif rule == "median_margin":
        med = float(np.median(vals))
        idx = int(np.argmin(np.abs(vals - med)))
        desc = f"Median Comp-A {condition} global-ref for {family}/{variant} ({vals[idx]:+.4f})"
    elif rule == "largest_alignet_shift":
        idx = int(np.argmax(np.abs(diffs)))
        desc = f"Largest |AligNet - UnAligNet| shift in {family} Comp-A {condition} ({diffs[idx]:+.4f})"
    else:
        raise ValueError(f"Unknown rule: {rule}")

    g, l = letters[int(gs[idx])], letters[int(ls[idx])]
    t = next(ch for ch in ("T", "A", "B", "C") if ch not in (g, l))
    return g, l, t, desc


def plot_single_example(
    global_letter: str = "H",
    local_letter: str = "S",
    third_letter: str = "T",
    condition: str = "s14",
    pos_idx: int = 0,
    representation: str = "pre_logits",
    selection_rule: str = "Predetermined example: C(H, S) with T=T",
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Plot similarities and global-minus-local margins for one selected compound."""
    store = load_workspace(str(out_dir) if out_dir else None)
    g, l, t = global_letter.upper(), local_letter.upper(), third_letter.upper()
    gi, li, ti = store["letter_to_i"][g], store["letter_to_i"][l], store["letter_to_i"][t]
    p_other = 1 if store["num_positions"] >= 2 else pos_idx

    thumb_specs = [
        (f"Ref C({g},{l})", f"cmp_{condition}_p{pos_idx}_{g}_{l}"),
        (f"I_glob({g})", f"iso_global_p{pos_idx}_{g}"),
        (f"I_glob({l})", f"iso_global_p{pos_idx}_{l}"),
        (f"I_loc({g})", f"iso_local_{condition}_p{pos_idx}_{g}"),
        (f"I_loc({l})", f"iso_local_{condition}_p{pos_idx}_{l}"),
        (f"C({g},{t}) p0", f"cmp_{condition}_p0_{g}_{t}"),
        (f"C({t},{l}) p0", f"cmp_{condition}_p0_{t}_{l}"),
        (f"C({g},{t}) p{p_other}", f"cmp_{condition}_p{p_other}_{g}_{t}"),
        (f"C({t},{l}) p{p_other}", f"cmp_{condition}_p{p_other}_{t}_{l}"),
    ]

    # Compute exact cosine similarities for all (family, variant, comparison_slice)
    scores: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for family in ("dino", "siglip"):
        for variant in VARIANT_ORDER:
            u = store["unit"][(family, variant, representation)]
            c_ref_pos = u[store["comp_idx"][(condition, pos_idx, gi, li)]]
            ig_g = u[store["iso_g_idx"][(pos_idx, gi)]]
            ig_l = u[store["iso_g_idx"][(pos_idx, li)]]
            il_g = u[store["iso_l_idx"][(condition, pos_idx, gi)]]
            il_l = u[store["iso_l_idx"][(condition, pos_idx, li)]]
            sg, sl = float(c_ref_pos @ ig_g), float(c_ref_pos @ ig_l)
            scores[(family, variant, "compA_global")] = (sg, sl, sg - sl)
            sg, sl = float(c_ref_pos @ il_g), float(c_ref_pos @ il_l)
            scores[(family, variant, "compA_local")] = (sg, sl, sg - sl)

            c_ref_p0 = u[store["comp_idx"][(condition, 0, gi, li)]]
            cg0 = u[store["comp_idx"][(condition, 0, gi, ti)]]
            cl0 = u[store["comp_idx"][(condition, 0, ti, li)]]
            sg, sl = float(c_ref_p0 @ cg0), float(c_ref_p0 @ cl0)
            scores[(family, variant, "compB_same")] = (sg, sl, sg - sl)

            cg1 = u[store["comp_idx"][(condition, p_other, gi, ti)]]
            cl1 = u[store["comp_idx"][(condition, p_other, ti, li)]]
            sg, sl = float(c_ref_p0 @ cg1), float(c_ref_p0 @ cl1)
            scores[(family, variant, "compB_cross")] = (sg, sl, sg - sl)

    width, height = 1480, 900
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (28, 16),
        f"2. Single-Example Breakdown — C({g}, {l}) with T={t}  |  rep={representation}, cond={condition}",
        _font(20, bold=True),
    )
    _draw_text(draw, (28, 44), f"Selection rule: {selection_rule}", _font(13), fill=TEXT_MUTED)
    _draw_legend(draw, width - 350, 20)

    # Thumbnail strip
    tsize = 122
    tx0, ty0, tgap = 28, 86, 22
    for idx, (label, img_id) in enumerate(thumb_specs):
        x = tx0 + idx * (tsize + tgap)
        _draw_text(draw, (x + tsize / 2, ty0 - 4), label, _font(12, bold=True), anchor="mb")
        tile = _load_preprocessed_tile(store, store["item_by_id"][img_id]["rel_path"], size=tsize)
        canvas.paste(tile, (x, ty0))
        draw.rectangle([x, ty0, x + tsize - 1, ty0 + tsize - 1], outline=(190, 190, 198))

    # Consistent x-axis bounds per comparison row
    row_defs = [
        (
            "Comparison A: Compound vs. Isolated Letters",
            [("compA_global", "Global ref scale (160px)"), ("compA_local", f"Local ref scale ({condition})")],
            245,
        ),
        (
            "Comparison B: Compound vs. Matching Compounds",
            [("compB_same", "Same position (p0 vs p0)"), ("compB_cross", f"Cross position (p0 vs p{p_other})")],
            575,
        ),
    ]

    for row_title, slices, y_top in row_defs:
        _draw_text(draw, (28, y_top), row_title, _font(16, bold=True))
        keys = [s[0] for s in slices]
        all_sims = [v for f in ("dino", "siglip") for var in VARIANT_ORDER for k in keys for v in scores[(f, var, k)][:2]]
        all_mars = [scores[(f, var, k)][2] for f in ("dino", "siglip") for var in VARIANT_ORDER for k in keys]
        sim_min = np.floor(min(all_sims) * 20) / 20 - 0.02
        sim_max = np.ceil(max(all_sims) * 20) / 20 + 0.02
        mar_lim = max(0.05, float(np.ceil(max(abs(m) for m in all_mars) * 20) / 20 + 0.01))

        # Marker legend for left subpanel
        _draw_text(
            draw,
            (width - 28, y_top + 2),
            "Filled circle = cos(ref, global match)   |   Hollow square = cos(ref, local match)",
            _font(12),
            fill=TEXT_MUTED,
            anchor="rt",
        )

        for f_idx, family in enumerate(("dino", "siglip")):
            px0 = 28 + f_idx * 725
            pw = 695
            py0 = y_top + 30
            ph = 270
            draw.rectangle([px0, py0, px0 + pw, py0 + ph], fill=BG_LIGHT, outline=GRID_LINE)
            _draw_text(draw, (px0 + 14, py0 + 10), f"{FAMILIES[family]} ({family})", _font(15, bold=True))

            sim_l, sim_r = px0 + 110, px0 + 380
            mar_l, mar_r = px0 + 425, px0 + pw - 22
            plot_t, plot_b = py0 + 48, py0 + ph - 38

            def map_sim(val: float) -> float:
                return sim_l + (val - sim_min) / max(sim_max - sim_min, 1e-6) * (sim_r - sim_l)

            def map_mar(val: float) -> float:
                return mar_l + (val + mar_lim) / (2 * mar_lim) * (mar_r - mar_l)

            _draw_text(draw, ((sim_l + sim_r) / 2, py0 + 30), "Cosine similarities", _font(12, bold=True), anchor="ma")
            _draw_text(draw, ((mar_l + mar_r) / 2, py0 + 30), "Margin (global - local)", _font(12, bold=True), anchor="ma")

            # Axis lines & ticks
            for t_val in np.linspace(sim_min, sim_max, 4):
                tx = map_sim(float(t_val))
                draw.line([tx, plot_t, tx, plot_b], fill=GRID_LINE, width=1)
                _draw_text(draw, (tx, plot_b + 6), f"{t_val:.2f}", _font(11), fill=TEXT_MUTED, anchor="ma")

            z_x = map_mar(0.0)
            for t_val in (-mar_lim, 0.0, mar_lim):
                tx = map_mar(float(t_val))
                draw.line([tx, plot_t, tx, plot_b], fill=TEXT_DARK if t_val == 0 else GRID_LINE, width=2 if t_val == 0 else 1)
                _draw_text(draw, (tx, plot_b + 6), f"{t_val:+.2f}", _font(11, bold=(t_val == 0)), fill=TEXT_DARK if t_val == 0 else TEXT_MUTED, anchor="ma")
            _draw_text(draw, (mar_l, plot_b + 22), "← Local", _font(11), fill=TEXT_MUTED, anchor="la")
            _draw_text(draw, (mar_r, plot_b + 22), "Global →", _font(11), fill=TEXT_MUTED, anchor="ra")

            # Rows for the two slices x 3 variants
            row_ys = np.linspace(plot_t + 16, plot_b - 16, 6)
            for s_idx, (s_key, s_label) in enumerate(slices):
                y_mid = float(np.mean(row_ys[s_idx * 3 : s_idx * 3 + 3]))
                _draw_text(draw, (px0 + 14, y_mid - 14), s_label.split("(")[0].strip(), _font(11, bold=True), fill=TEXT_DARK)
                _draw_text(draw, (px0 + 14, y_mid + 2), "(" + s_label.split("(")[1], _font(11), fill=TEXT_MUTED)
                if s_idx == 0:
                    sep_y = float((row_ys[2] + row_ys[3]) / 2)
                    draw.line([px0 + 10, sep_y, px0 + pw - 10, sep_y], fill=GRID_LINE, width=1)

                for v_idx, var in enumerate(VARIANT_ORDER):
                    ry = float(row_ys[s_idx * 3 + v_idx])
                    col = VARIANT_COLORS[var]
                    sg, sl, mar = scores[(family, var, s_key)]

                    # Dot plot of sim_g and sim_l
                    xg, xl = map_sim(sg), map_sim(sl)
                    draw.line([min(xg, xl), ry, max(xg, xl), ry], fill=col, width=2)
                    draw.rectangle([xl - 4, ry - 4, xl + 4, ry + 4], fill="white", outline=col, width=2)
                    draw.ellipse([xg - 5, ry - 5, xg + 5, ry + 5], fill=col)

                    # Margin bar from zero line
                    xm = map_mar(mar)
                    draw.rectangle([min(z_x, xm), ry - 5, max(z_x, xm), ry + 5], fill=col)
                    _draw_text(draw, (px0 + pw - 22, ry), "", _font(10))

    return canvas


# ---------------------------------------------------------------------------
# 3. What happens across the dataset?
# ---------------------------------------------------------------------------
def _cluster_boot_ci(pair_diffs: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(pair_diffs)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = pair_diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(pair_diffs.mean()), float(lo), float(hi)


def plot_dataset_overview(
    representation: str = "pre_logits",
    comp_a_scale: str = "global",
    comp_b_regime: str = "cross_pos",
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Plot dataset margin distributions, choice fractions, and UnAligNet-vs-AligNet paired scatter."""
    store = load_workspace(str(out_dir) if out_dir else None)

    width, height = 1480, 820
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (28, 16),
        f"3. Dataset-Wide Margin Distributions & Paired Comparison  (rep={representation})",
        _font(20, bold=True),
    )
    _draw_text(
        draw,
        (28, 44),
        f"Left of each panel: trial margin distributions + choice % (Comp A={comp_a_scale} ref; Comp B={comp_b_regime}). "
        "Right: paired 325 letter-pair means {G, L}.",
        _font(13),
        fill=TEXT_MUTED,
    )
    _draw_legend(draw, width - 350, 20)

    panels = [
        ("comparison_a", f"Comparison A (Isolated ref: {comp_a_scale})", 84),
        ("comparison_b", f"Comparison B (Compound ref: {comp_b_regime})", 452),
    ]

    for comp_name, row_label, y_top in panels:
        _draw_text(draw, (28, y_top), row_label, _font(15, bold=True))

        # Collect trial margins & pair means for both families to share axes within row
        fam_data = {}
        all_vals = []
        all_pairs_vals = []
        for family in ("dino", "siglip"):
            fam_data[family] = {}
            for var in VARIANT_ORDER:
                if comp_name == "comparison_a":
                    m_trials, pids, _ = _comp_a_pair_matrix(store, family, var, representation, ref_scales=(comp_a_scale,))
                else:
                    m_trials, pids, _ = _comp_b_pair_matrix(store, family, var, representation, regimes=(comp_b_regime,))
                p_means = _unordered_pair_means(m_trials, pids, len(store["pairs"]))
                fam_data[family][var] = (m_trials, p_means)
                all_vals.append(m_trials)
                if var in ("unalignet", "alignet"):
                    all_pairs_vals.append(p_means)

        stacked = np.concatenate(all_vals)
        lo_x = float(np.percentile(stacked, 0.5))
        hi_x = float(np.percentile(stacked, 99.5))
        span = max(abs(lo_x), abs(hi_x), 0.04) * 1.15
        x_min, x_max = -span, span

        p_stacked = np.concatenate(all_pairs_vals)
        p_span = max(abs(float(p_stacked.min())), abs(float(p_stacked.max())), 0.03) * 1.15

        for f_idx, family in enumerate(("dino", "siglip")):
            px0 = 28 + f_idx * 725
            pw, ph = 695, 325
            py0 = y_top + 26
            draw.rectangle([px0, py0, px0 + pw, py0 + ph], fill=BG_LIGHT, outline=GRID_LINE)
            _draw_text(draw, (px0 + 14, py0 + 10), f"{FAMILIES[family]} ({family})", _font(15, bold=True))

            # Left half: Distributions
            dl, dr = px0 + 92, px0 + 365
            dt, db = py0 + 46, py0 + ph - 52
            _draw_text(draw, ((dl + dr) / 2, py0 + 28), "Trial margin distribution", _font(12, bold=True), anchor="ma")

            def map_d(v: float) -> float:
                return dl + np.clip((v - x_min) / (x_max - x_min), 0.0, 1.0) * (dr - dl)

            for tv in (-x_max * 0.75, 0.0, x_max * 0.75):
                tx = map_d(tv)
                draw.line([tx, dt, tx, db], fill=TEXT_DARK if tv == 0 else GRID_LINE, width=2 if tv == 0 else 1)
                _draw_text(draw, (tx, db + 5), f"{tv:+.2f}", _font(11, bold=(tv == 0)), fill=TEXT_DARK if tv == 0 else TEXT_MUTED, anchor="ma")
            _draw_text(draw, (dl, db + 24), "← Local closer", _font(11), fill=TEXT_MUTED, anchor="la")
            _draw_text(draw, (dr, db + 24), "Global closer →", _font(11), fill=TEXT_MUTED, anchor="ra")

            bins = np.linspace(x_min, x_max, 45)
            centers = 0.5 * (bins[:-1] + bins[1:])
            band_h = (db - dt) / 3
            for v_idx, var in enumerate(VARIANT_ORDER):
                m_trials, _ = fam_data[family][var]
                col = VARIANT_COLORS[var]
                base_y = dt + (v_idx + 0.72) * band_h
                _draw_text(draw, (px0 + 14, base_y - 16), VARIANT_LABELS[var], _font(12, bold=True), fill=col)

                hist, _ = np.histogram(m_trials, bins=bins, density=True)
                if hist.max() > 0:
                    hist_scaled = hist / hist.max() * (band_h * 0.52)
                else:
                    hist_scaled = hist
                poly = [(map_d(float(centers[0])), base_y)]
                for cx, hval in zip(centers, hist_scaled):
                    poly.append((map_d(float(cx)), base_y - float(hval)))
                poly.append((map_d(float(centers[-1])), base_y))
                light_col = tuple(min(255, int(c * 0.35 + 255 * 0.65)) for c in col)
                draw.polygon(poly, fill=light_col, outline=col)

                q25, q50, q75 = np.percentile(m_trials, [25, 50, 75])
                draw.line([map_d(float(q25)), base_y, map_d(float(q75)), base_y], fill=col, width=3)
                mx = map_d(float(q50))
                draw.ellipse([mx - 4, base_y - 4, mx + 4, base_y + 4], fill=col, outline="white")

                g_pct = 100.0 * float(np.mean(m_trials > 0))
                l_pct = 100.0 * float(np.mean(m_trials < 0))
                t_pct = 100.0 * float(np.mean(m_trials == 0))
                _draw_text(
                    draw,
                    (px0 + 14, base_y + 2),
                    f"G:{g_pct:.0f}% L:{l_pct:.0f}% T:{t_pct:.0f}%",
                    _font(10),
                    fill=TEXT_MUTED,
                )

            # Right half: Paired scatterplot UnAligNet vs AligNet
            sl, sr = px0 + 448, px0 + pw - 22
            st, sb = py0 + 46, py0 + ph - 52
            side = min(sr - sl, sb - st)
            sr = sl + side
            sb = st + side
            _draw_text(draw, ((sl + sr) / 2, py0 + 28), "Paired: UnAligNet (x) vs. AligNet (y)", _font(11, bold=True), anchor="ma")
            draw.rectangle([sl, st, sr, sb], fill="white", outline=AXIS_LINE)

            def map_sx(v: float) -> float:
                return sl + np.clip((v + p_span) / (2 * p_span), 0.0, 1.0) * (sr - sl)

            def map_sy(v: float) -> float:
                return sb - np.clip((v + p_span) / (2 * p_span), 0.0, 1.0) * (sb - st)

            # Zero axes and identity line y = x
            zx, zy = map_sx(0.0), map_sy(0.0)
            _dashed_line(draw, (zx, st), (zx, sb), fill=(180, 180, 188), width=1)
            _dashed_line(draw, (sl, zy), (sr, zy), fill=(180, 180, 188), width=1)
            draw.line([sl, sb, sr, st], fill=TEXT_DARK, width=2)

            _, p_un = fam_data[family]["unalignet"]
            _, p_al = fam_data[family]["alignet"]
            for xv, yv in zip(p_un, p_al):
                px, py = map_sx(float(xv)), map_sy(float(yv))
                dot_col = (37, 99, 235) if yv < xv else (217, 119, 6)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=dot_col)

            d_mean, ci_lo, ci_hi = _cluster_boot_ci(p_al - p_un)
            below_pct = 100.0 * float(np.mean(p_al < p_un))
            _draw_text(draw, (sl + 6, st + 6), f"Δ(Alig-Un) = {d_mean:+.4f}", _font(11, bold=True), fill=TEXT_DARK)
            _draw_text(draw, (sl + 6, st + 20), f"95% pair CI [{ci_lo:+.4f}, {ci_hi:+.4f}]", _font(10), fill=TEXT_MUTED)
            _draw_text(draw, (sl + 6, st + 34), f"{below_pct:.1f}% pairs below y=x", _font(10), fill=TEXT_MUTED)

            _draw_text(draw, ((sl + sr) / 2, sb + 18), "UnAligNet margin (pair mean)", _font(11), fill=TEXT_DARK, anchor="ma")
            _draw_text(draw, (sl, sb + 4), f"-{p_span:.2f}", _font(10), fill=TEXT_MUTED, anchor="la")
            _draw_text(draw, (sr, sb + 4), f"+{p_span:.2f}", _font(10), fill=TEXT_MUTED, anchor="ra")
            _draw_text(draw, (sl - 6, st), f"+{p_span:.2f}", _font(10), fill=TEXT_MUTED, anchor="ra")
            _draw_text(draw, (sl - 6, sb - 10), f"-{p_span:.2f}", _font(10), fill=TEXT_MUTED, anchor="ra")

    return canvas


# ---------------------------------------------------------------------------
# 4. Which conditions or letter combinations matter?
# ---------------------------------------------------------------------------
def _diverging_rgb(val: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(val):
        return (215, 215, 220)
    t = float(np.clip(val / max(vmax, 1e-8), -1.0, 1.0))
    if t < 0:
        a = -t
        return (int(255 - a * 200), int(255 - a * 150), 255)  # blue for negative
    a = t
    return (255, int(255 - a * 185), int(255 - a * 185))      # red for positive


def plot_letter_heatmap(
    representation: str = "pre_logits",
    comparison: str = "comparison_a",
    subcondition: str = "ALL|global",
    metric: str = "alignet_minus_unalignet",
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Plot 26x26 Global vs. Local letter heatmaps for margin or AligNet-UnAligNet shift."""
    store = load_workspace(str(out_dir) if out_dir else None)
    letters = store["letters"]
    k = len(letters)
    cond_part, slice_part = subcondition.split("|")
    conds = store["conditions"] if cond_part == "ALL" else [cond_part]

    def get_mat(fam: str, var: str) -> np.ndarray:
        if comparison == "comparison_a":
            _, _, mat = _comp_a_pair_matrix(store, fam, var, representation, conds=conds, ref_scales=(slice_part,))
        else:
            _, _, mat = _comp_b_pair_matrix(store, fam, var, representation, conds=conds, regimes=(slice_part,))
        return mat

    mats = {}
    for fam in ("dino", "siglip"):
        if metric == "alignet_minus_unalignet":
            mats[fam] = get_mat(fam, "alignet") - get_mat(fam, "unalignet")
        elif metric == "alignet_minus_base":
            mats[fam] = get_mat(fam, "alignet") - get_mat(fam, "base")
        elif metric == "unalignet_minus_base":
            mats[fam] = get_mat(fam, "unalignet") - get_mat(fam, "base")
        elif metric.startswith("margin_"):
            mats[fam] = get_mat(fam, metric.split("_", 1)[1])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    finite_vals = np.concatenate([m[np.isfinite(m)] for m in mats.values()])
    vmax = max(0.01, float(np.percentile(np.abs(finite_vals), 99.0)))

    cell = 20
    grid_px = k * cell
    width, height = 1320, 690
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (28, 16),
        f"4a. Global vs. Local Letter Matrix — {metric} ({comparison}: {subcondition}, rep={representation})",
        _font(18, bold=True),
    )
    _draw_text(
        draw,
        (28, 44),
        "Rows = Global letter G, Columns = Local letter L. Blue (< 0) = more local / shift toward local; Red (> 0) = more global / shift toward global.",
        _font(13),
        fill=TEXT_MUTED,
    )

    for f_idx, fam in enumerate(("dino", "siglip")):
        ox = 68 + f_idx * 580
        oy = 98
        _draw_text(draw, (ox + grid_px / 2, oy - 24), f"{FAMILIES[fam]} ({fam})", _font(15, bold=True), anchor="ma")
        mat = mats[fam]
        for gi, gch in enumerate(letters):
            _draw_text(draw, (ox - 8, oy + gi * cell + cell / 2), gch, _font(11, bold=True), fill=TEXT_DARK, anchor="rm")
            for li, lch in enumerate(letters):
                if gi == 0:
                    _draw_text(draw, (ox + li * cell + cell / 2, oy + grid_px + 8), lch, _font(11, bold=True), fill=TEXT_DARK, anchor="ma")
                x0, y0 = ox + li * cell, oy + gi * cell
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=_diverging_rgb(float(mat[gi, li]), vmax))
        draw.rectangle([ox, oy, ox + grid_px, oy + grid_px], outline=TEXT_DARK, width=1)
        _draw_text(draw, (ox + grid_px / 2, oy + grid_px + 28), "Local letter L", _font(13, bold=True), anchor="ma")
        _draw_text(draw, (ox - 34, oy + grid_px / 2), "G", _font(13, bold=True), anchor="rm")

    # Shared colorbar
    cb_x, cb_y, cb_w, cb_h = 1235, 110, 22, 490
    for py in range(cb_h):
        val = vmax * (1.0 - 2.0 * py / max(cb_h - 1, 1))
        draw.line([cb_x, cb_y + py, cb_x + cb_w, cb_y + py], fill=_diverging_rgb(val, vmax))
    draw.rectangle([cb_x, cb_y, cb_x + cb_w, cb_y + cb_h], outline=TEXT_DARK)
    _draw_text(draw, (cb_x + cb_w + 8, cb_y), f"+{vmax:.3f}", _font(12, bold=True))
    _draw_text(draw, (cb_x + cb_w + 8, cb_y + cb_h / 2), " 0.000", _font(12, bold=True), anchor="lm")
    _draw_text(draw, (cb_x + cb_w + 8, cb_y + cb_h), f"-{vmax:.3f}", _font(12, bold=True), anchor="lb")

    return canvas


def plot_condition_breakdown(representation: str = "pre_logits", out_dir: str | Path | None = None) -> Image.Image:
    """Compare element size/spacing (s10/s14/s18), reference scale, and position across variants."""
    store = load_workspace(str(out_dir) if out_dir else None)
    conds = store["conditions"]

    # Collect (mean, lo, hi) per (family, var, comp, slice_name)
    stats: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    all_vals = []
    comp_a_slices = [f"{c}|{s}" for s in ("global", "local") for c in conds]
    comp_b_slices = [f"{c}|{r}" for r in ("same_pos", "cross_pos") for c in conds]

    for fam in ("dino", "siglip"):
        for var in VARIANT_ORDER:
            for sub in comp_a_slices:
                c, s = sub.split("|")
                m_trials, pids, _ = _comp_a_pair_matrix(store, fam, var, representation, conds=[c], ref_scales=(s,))
                p_means = _unordered_pair_means(m_trials, pids, len(store["pairs"]))
                mean, lo, hi = _cluster_boot_ci(p_means, n_boot=400, seed=0)
                stats[(fam, var, "comparison_a", sub)] = (mean, lo, hi)
                all_vals.extend([lo, hi])
            for sub in comp_b_slices:
                c, r = sub.split("|")
                m_trials, pids, _ = _comp_b_pair_matrix(store, fam, var, representation, conds=[c], regimes=(r,))
                p_means = _unordered_pair_means(m_trials, pids, len(store["pairs"]))
                mean, lo, hi = _cluster_boot_ci(p_means, n_boot=400, seed=0)
                stats[(fam, var, "comparison_b", sub)] = (mean, lo, hi)
                all_vals.extend([lo, hi])

    y_span = max(0.05, float(max(abs(v) for v in all_vals)) * 1.15)
    width, height = 1480, 740
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (28, 16),
        f"4b. Effect of Local Size/Spacing, Reference Scale, and Position  (rep={representation})",
        _font(20, bold=True),
    )
    _draw_text(
        draw,
        (28, 44),
        "Markers show mean margin with 95% pair-cluster CIs across conditions in manifest.json. Rotation is fixed at 0° across all stimuli.",
        _font(13),
        fill=TEXT_MUTED,
    )
    _draw_legend(draw, width - 350, 20)

    col_defs = [
        ("comparison_a", "Comparison A: Element size/spacing × Isolated reference scale", comp_a_slices, 28),
        ("comparison_b", "Comparison B: Element size/spacing × Candidate position", comp_b_slices, 755),
    ]

    for comp_name, title, slices, x_left in col_defs:
        pw = 695
        _draw_text(draw, (x_left, 82), title, _font(15, bold=True))
        for f_idx, fam in enumerate(("dino", "siglip")):
            py0 = 112 + f_idx * 305
            ph = 278
            draw.rectangle([x_left, py0, x_left + pw, py0 + ph], fill=BG_LIGHT, outline=GRID_LINE)
            _draw_text(draw, (x_left + 14, py0 + 10), f"{FAMILIES[fam]} ({fam})", _font(14, bold=True))

            pl, pr = x_left + 72, x_left + pw - 22
            pt, pb = py0 + 38, py0 + ph - 48

            def map_y(v: float) -> float:
                return pb - (v + y_span) / (2 * y_span) * (pb - pt)

            for tv in (-y_span * 0.75, 0.0, y_span * 0.75):
                ty = map_y(tv)
                draw.line([pl, ty, pr, ty], fill=TEXT_DARK if tv == 0 else GRID_LINE, width=2 if tv == 0 else 1)
                _draw_text(draw, (pl - 6, ty), f"{tv:+.2f}", _font(11, bold=(tv == 0)), fill=TEXT_DARK if tv == 0 else TEXT_MUTED, anchor="rm")

            group_w = (pr - pl) / len(slices)
            mid_sep = pl + 3 * group_w
            _dashed_line(draw, (mid_sep, pt), (mid_sep, pb), fill=(180, 180, 188), width=1)

            for s_idx, sub in enumerate(slices):
                gx = pl + (s_idx + 0.5) * group_w
                c_part, r_part = sub.split("|")
                _draw_text(draw, (gx, pb + 6), c_part, _font(12, bold=True), anchor="ma")
                _draw_text(draw, (gx, pb + 22), r_part, _font(11), fill=TEXT_MUTED, anchor="ma")
                for v_idx, var in enumerate(VARIANT_ORDER):
                    vx = gx + (v_idx - 1) * 14
                    mean, lo, hi = stats[(fam, var, comp_name, sub)]
                    col = VARIANT_COLORS[var]
                    draw.line([vx, map_y(lo), vx, map_y(hi)], fill=col, width=2)
                    ym = map_y(mean)
                    draw.ellipse([vx - 4, ym - 4, vx + 4, ym + 4], fill=col, outline="white")

    return canvas


def show_cell_gallery(
    representation: str = "pre_logits",
    family: str = "siglip",
    comparison: str = "comparison_a",
    condition: str = "s14",
    mode: str = "extremes_and_typical",
    custom_pairs: list[tuple[str, str]] | None = None,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Render an inspection gallery of selected (G, L) cells with explicit selection labels."""
    store = load_workspace(str(out_dir) if out_dir else None)
    letters = store["letters"]
    k = len(letters)

    def get_m(var: str) -> np.ndarray:
        if comparison == "comparison_a":
            return _comp_a_pair_matrix(store, family, var, representation, conds=[condition], ref_scales=("global",))[2]
        return _comp_b_pair_matrix(store, family, var, representation, conds=[condition], regimes=("cross_pos",))[2]

    m_base = get_m("base")
    m_un = get_m("unalignet")
    m_al = get_m("alignet")
    shift = m_al - m_un

    gs, ls = np.where(~np.eye(k, dtype=bool))
    al_vals = m_al[gs, ls]
    sh_vals = shift[gs, ls]

    chosen: list[tuple[str, int, int]] = []
    if mode == "custom" and custom_pairs:
        for g, l in custom_pairs[:5]:
            chosen.append(("Manual selection", store["letter_to_i"][g.upper()], store["letter_to_i"][l.upper()]))
    elif mode == "random":
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(gs), size=min(5, len(gs)), replace=False)
        for p in picks:
            chosen.append((f"Random sample (seed={seed})", int(gs[p]), int(ls[p])))
    else:
        picks = [
            ("Extreme: most global", int(np.argmax(al_vals))),
            ("Typical: median margin", int(np.argmin(np.abs(al_vals - float(np.median(al_vals)))))),
            ("Extreme: most local", int(np.argmin(al_vals))),
            ("Extreme: max -shift", int(np.argmin(sh_vals))),
            ("Extreme: max +shift", int(np.argmax(sh_vals))),
        ]
        for label, idx in picks:
            chosen.append((label, int(gs[idx]), int(ls[idx])))

    tile_s = 210
    pad_x, gap_x = 28, 22
    width = pad_x * 2 + len(chosen) * tile_s + (len(chosen) - 1) * gap_x
    height = 375
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (pad_x, 16),
        f"4c. Selected Cell Inspection ({FAMILIES[family]} | {comparison} | cond={condition} | rep={representation} | mode={mode})",
        _font(17, bold=True),
    )

    for idx, (label, gi, li) in enumerate(chosen):
        g, l = letters[gi], letters[li]
        x0 = pad_x + idx * (tile_s + gap_x)
        y0 = 76
        _draw_text(draw, (x0 + tile_s / 2, y0 - 22), label, _font(12, bold=True), fill=TEXT_DARK, anchor="mb")
        _draw_text(draw, (x0 + tile_s / 2, y0 - 6), f"C({g}, {l})", _font(13, bold=True), fill=VARIANT_COLORS["alignet"], anchor="mb")
        rec = store["item_by_id"][f"cmp_{condition}_p0_{g}_{l}"]
        canvas.paste(_load_preprocessed_tile(store, rec["rel_path"], size=tile_s), (x0, y0))
        draw.rectangle([x0, y0, x0 + tile_s - 1, y0 + tile_s - 1], outline=(180, 180, 188))

        lines = [
            f"Base: {m_base[gi, li]:+.4f}",
            f"UnAligNet: {m_un[gi, li]:+.4f}",
            f"AligNet: {m_al[gi, li]:+.4f}",
            f"Alig - Un: {shift[gi, li]:+.4f}",
        ]
        for li_t, text in enumerate(lines):
            _draw_text(draw, (x0 + tile_s / 2, y0 + tile_s + 8 + li_t * 16), text, _font(12, bold=(li_t == 3)), anchor="ma")

    return canvas


# ---------------------------------------------------------------------------
# 5. Check the letter-substitution issue
# ---------------------------------------------------------------------------
def plot_substitution_check(
    representation: str = "pre_logits",
    condition: str = "s14",
    pos_idx: int = 0,
    example_x: str = "H",
    example_y: str = "S",
    example_z: str = "T",
    out_dir: str | Path | None = None,
) -> Image.Image:
    """Analyze third-letter resemblance in Comp B and matched global-vs-local substitution."""
    store = load_workspace(str(out_dir) if out_dir else None)
    letters = store["letters"]
    k = len(letters)
    x_ch, y_ch, z_ch = example_x.upper(), example_y.upper(), example_z.upper()
    cfg = store["manifest"]["config"]
    shift_xy = cfg["shifts"][pos_idx]

    width, height = 1480, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    _draw_text(
        draw,
        (28, 16),
        f"5. Letter-Substitution Control & Third-Letter Resemblance  (cond={condition}, rep={representation})",
        _font(20, bold=True),
    )
    _draw_text(
        draw,
        (28, 44),
        f"All four comparison images share exact pos_idx={pos_idx} shift=({shift_xy[0]},{shift_xy[1]}) and rotation={cfg['rotation_deg']}°, isolating identity swap {x_ch} ↔ {y_ch} with {z_ch} fixed.",
        _font(13),
        fill=TEXT_MUTED,
    )
    _draw_legend(draw, width - 350, 20)

    # Top strip: 4 matched stimuli showing global vs local substitution
    tsize = 148
    diagram = [
        (f"Global sub: C({x_ch}, {z_ch})", f"cmp_{condition}_p{pos_idx}_{x_ch}_{z_ch}", 28),
        (f"Global sub: C({y_ch}, {z_ch})", f"cmp_{condition}_p{pos_idx}_{y_ch}_{z_ch}", 28 + tsize + 18),
        (f"Local sub: C({z_ch}, {x_ch})", f"cmp_{condition}_p{pos_idx}_{z_ch}_{x_ch}", 420),
        (f"Local sub: C({z_ch}, {y_ch})", f"cmp_{condition}_p{pos_idx}_{z_ch}_{y_ch}", 420 + tsize + 18),
    ]
    _draw_text(draw, (28, 76), f"Swap {x_ch} ↔ {y_ch} globally (local {z_ch} fixed)", _font(13, bold=True))
    _draw_text(draw, (420, 76), f"Swap {x_ch} ↔ {y_ch} locally (global {z_ch} fixed)", _font(13, bold=True))
    for cap, img_id, x0 in diagram:
        y0 = 102
        canvas.paste(_load_preprocessed_tile(store, store["item_by_id"][img_id]["rel_path"], size=tsize), (x0, y0))
        draw.rectangle([x0, y0, x0 + tsize - 1, y0 + tsize - 1], outline=(180, 180, 188))
        _draw_text(draw, (x0 + tsize / 2, y0 + tsize + 6), cap, _font(12, bold=True), anchor="ma")

    # Right side of top strip: explanation card
    draw.rectangle([780, 96, width - 28, 268], fill=BG_LIGHT, outline=GRID_LINE)
    notes = [
        "Why check matched substitution?",
        f"• In Comp B: cos(C(G,L), C(G,T)) - cos(C(G,L), C(T,L)), the third letter T can resemble G or L.",
        f"• Matched substitution averages over all 24 unchanged letters Z ≠ {x_ch},{y_ch} for each of 325 pairs {{X, Y}}:",
        "    S_preserve_global = mean_Z cos(C(Z, X), C(Z, Y))   (local X ↔ Y swapped)",
        "    S_preserve_local  = mean_Z cos(C(X, Z), C(Y, Z))   (global X ↔ Y swapped)",
        "• Δ_sub = S_preserve_global - S_preserve_local (> 0 means preserving global Z keeps images closer).",
    ]
    for ni, line in enumerate(notes):
        _draw_text(draw, (796, 108 + ni * 24), line, _font(12, bold=(ni == 0)), fill=TEXT_DARK if ni == 0 else TEXT_MUTED)

    # Bottom-left: Third-letter resemblance confound in Comparison B (AligNet across both families)
    bl_x, bl_y, bl_w, bl_h = 28, 305, 695, 425
    draw.rectangle([bl_x, bl_y, bl_x + bl_w, bl_y + bl_h], fill=BG_LIGHT, outline=GRID_LINE)
    _draw_text(draw, (bl_x + 14, bl_y + 10), "Comp-B margin (y) vs. third-letter T resemblance (x) [AligNet]", _font(14, bold=True))

    gg, ll, tt = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")
    valid = (gg != ll) & (tt != gg) & (tt != ll)
    vg, vl, vt = gg[valid], ll[valid], tt[valid]

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(vg), size=1200, replace=False)

    for f_idx, fam in enumerate(("dino", "siglip")):
        sx0 = bl_x + 20 + f_idx * 340
        sy0 = bl_y + 48
        sw, sh = 315, 325
        sl, sr = sx0 + 48, sx0 + sw - 12
        st, sb = sy0 + 28, sy0 + sh - 38
        draw.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], fill="white", outline=GRID_LINE)

        u = store["unit"][(fam, "alignet", representation)]
        ref_ids = np.array([store["comp_idx"][(condition, pos_idx, g, l)] for g in range(k) for l in range(k)])
        m4 = (u[ref_ids] @ u[ref_ids].T).reshape(k, k, k, k)
        mar_triad = m4[vg, vl, vg, vt] - m4[vg, vl, vt, vl]

        iso_ids = np.array([store["iso_g_idx"][(pos_idx, x)] for x in range(k)])
        iso_sim = u[iso_ids] @ u[iso_ids].T
        t_asym = iso_sim[vt, vl] - iso_sim[vt, vg]  # positive => T resembles L more than G

        r_val = float(np.corrcoef(t_asym, mar_triad)[0, 1])
        _draw_text(draw, ((sl + sr) / 2, sy0 + 6), f"{FAMILIES[fam]} (r = {r_val:+.3f})", _font(13, bold=True), anchor="ma")

        xm = max(0.05, float(np.percentile(np.abs(t_asym), 99.0)) * 1.1)
        y_lo, y_hi = float(np.percentile(mar_triad, 1.0)), float(np.percentile(mar_triad, 99.0))
        y_pad = (y_hi - y_lo) * 0.15
        y_lo, y_hi = min(y_lo - y_pad, -0.02), max(y_hi + y_pad, 0.02)

        def mx(v: float) -> float:
            return sl + np.clip((v + xm) / (2 * xm), 0.0, 1.0) * (sr - sl)

        def my(v: float) -> float:
            return sb - np.clip((v - y_lo) / (y_hi - y_lo), 0.0, 1.0) * (sb - st)

        _dashed_line(draw, (mx(0.0), st), (mx(0.0), sb), fill=AXIS_LINE)
        _dashed_line(draw, (sl, my(0.0)), (sr, my(0.0)), fill=AXIS_LINE)

        for idx in sample_idx:
            px, py = mx(float(t_asym[idx])), my(float(mar_triad[idx]))
            draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=VARIANT_COLORS["alignet"])

        slope, intercept = np.polyfit(t_asym, mar_triad, 1)
        draw.line([mx(-xm), my(float(slope * (-xm) + intercept)), mx(xm), my(float(slope * xm + intercept))], fill=(220, 38, 38), width=2)
        _draw_text(draw, ((sl + sr) / 2, sb + 12), "cos(I(T),I(L)) - cos(I(T),I(G))", _font(10), fill=TEXT_MUTED, anchor="ma")
        _draw_text(draw, (sl - 4, st), f"{y_hi:+.2f}", _font(10), fill=TEXT_MUTED, anchor="rm")
        _draw_text(draw, (sl - 4, sb), f"{y_lo:+.2f}", _font(10), fill=TEXT_MUTED, anchor="rm")

    # Bottom-right: Matched substitution comparison across 325 pairs {X, Y}
    br_x, br_y, br_w, br_h = 755, 305, 695, 425
    draw.rectangle([br_x, br_y, br_x + br_w, br_y + br_h], fill=BG_LIGHT, outline=GRID_LINE)
    _draw_text(draw, (br_x + 14, br_y + 10), "Matched global vs. local substitution across 325 pairs {X, Y}", _font(14, bold=True))

    sub_stats: dict[tuple[str, str], tuple[float, float, float, float, float]] = {}
    all_subs = []
    for fam in ("dino", "siglip"):
        for var in VARIANT_ORDER:
            u = store["unit"][(fam, var, representation)]
            ref_ids = np.array([store["comp_idx"][(condition, pos_idx, g, l)] for g in range(k) for l in range(k)])
            m4 = (u[ref_ids] @ u[ref_ids].T).reshape(k, k, k, k)
            s_pg = np.empty(len(store["pairs"]), dtype=np.float64)
            s_pl = np.empty(len(store["pairs"]), dtype=np.float64)
            for pid, (xi, yi) in enumerate(store["pairs"]):
                zs = [zi for zi in range(k) if zi not in (xi, yi)]
                s_pg[pid] = float(np.mean([m4[zi, xi, zi, yi] for zi in zs]))  # global Z fixed, local X<->Y
                s_pl[pid] = float(np.mean([m4[xi, zi, yi, zi] for zi in zs]))  # local Z fixed, global X<->Y
            diff = s_pg - s_pl
            mean_d, lo_d, hi_d = _cluster_boot_ci(diff, n_boot=600, seed=0)
            sub_stats[(fam, var)] = (float(s_pg.mean()), float(s_pl.mean()), mean_d, lo_d, hi_d)
            all_subs.extend([lo_d, hi_d])

    s_span = max(0.05, float(max(abs(v) for v in all_subs)) * 1.2)
    sl, sr = br_x + 190, br_x + br_w - 24
    st, sb = br_y + 52, br_y + br_h - 46
    draw.rectangle([sl, st, sr, sb], fill="white", outline=GRID_LINE)

    def m_sub(v: float) -> float:
        return sl + (v + s_span) / (2 * s_span) * (sr - sl)

    zx = m_sub(0.0)
    for tv in (-s_span * 0.75, 0.0, s_span * 0.75):
        tx = m_sub(tv)
        draw.line([tx, st, tx, sb], fill=TEXT_DARK if tv == 0 else GRID_LINE, width=2 if tv == 0 else 1)
        _draw_text(draw, (tx, sb + 6), f"{tv:+.2f}", _font(11, bold=(tv == 0)), fill=TEXT_DARK if tv == 0 else TEXT_MUTED, anchor="ma")
    _draw_text(draw, (sl, sb + 24), "← Local match preserved more", _font(11), fill=TEXT_MUTED, anchor="la")
    _draw_text(draw, (sr, sb + 24), "Global match preserved more →", _font(11), fill=TEXT_MUTED, anchor="ra")

    row_ys = np.linspace(st + 32, sb - 32, 6)
    for f_idx, fam in enumerate(("dino", "siglip")):
        _draw_text(draw, (br_x + 14, float(np.mean(row_ys[f_idx * 3 : f_idx * 3 + 3]))), FAMILIES[fam], _font(13, bold=True), anchor="lm")
        if f_idx == 0:
            sep = float((row_ys[2] + row_ys[3]) / 2)
            draw.line([br_x + 12, sep, br_x + br_w - 12, sep], fill=GRID_LINE)
        for v_idx, var in enumerate(VARIANT_ORDER):
            ry = float(row_ys[f_idx * 3 + v_idx])
            col = VARIANT_COLORS[var]
            s_pg, s_pl, mean_d, lo_d, hi_d = sub_stats[(fam, var)]
            _draw_text(draw, (sl - 10, ry), VARIANT_LABELS[var], _font(12, bold=True), fill=col, anchor="rm")
            draw.rectangle([min(zx, m_sub(mean_d)), ry - 7, max(zx, m_sub(mean_d)), ry + 7], fill=col)
            draw.line([m_sub(lo_d), ry, m_sub(hi_d), ry], fill=TEXT_DARK, width=2)
            label_x = zx + 10 if mean_d <= 0 else zx - 10
            label_anc = "lm" if mean_d <= 0 else "rm"
            _draw_text(
                draw,
                (label_x, ry),
                f"Δ={mean_d:+.4f} (pg={s_pg:.3f}, pl={s_pl:.3f})",
                _font(11),
                fill=TEXT_DARK,
                anchor=label_anc,
            )

    return canvas


def print_summary_tables(representation: str = "pre_logits", out_dir: str | Path | None = None) -> None:
    """Print detailed condition-level summary and paired contrast tables for optional inspection."""
    store = load_workspace(str(out_dir) if out_dir else None)
    print(f"=== Detailed Summary ({representation}) ===")
    for r in store["summary"]["summary"]:
        if r["representation"] != representation:
            continue
        print(
            f"{r['family']:6s} | {r['variant']:9s} | {r['comparison']:12s} | {r['subcondition']:14s} | "
            f"mean={r['mean_margin']:+.4f} | glob={r['global_frac']*100:5.1f}% | loc={r['local_frac']*100:5.1f}% | tie={r['tie_frac']*100:4.1f}%"
        )
    print(f"\n=== Paired Contrasts ({representation}, 95% pair-cluster bootstrap CIs) ===")
    for r in store["summary"]["paired_contrasts"]:
        if r["representation"] != representation:
            continue
        print(
            f"{r['family']:6s} | {r['contrast']:21s} | {r['comparison']:12s} | {r['subcondition']:14s} | "
            f"d_margin={r['delta_margin']:+.4f} [{r['margin_ci95_low']:+.4f}, {r['margin_ci95_high']:+.4f}] | "
            f"d_glob={r['delta_global_frac']:+.4f}"
        )
