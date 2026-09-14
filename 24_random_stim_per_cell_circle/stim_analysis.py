"""Helpers for per-cell *circular* stim-response analysis.

Companion to the patch-based analysis in experiment 23. The stim region
here is a small circular disk at an arbitrary pixel inside the cell —
no grid alignment, no patch bounding box. The per-region descriptor is
therefore ``(center_y, center_x, radius)`` rather than
``(y_min, x_min, patch_size)``.

What changes vs the patch version:

  * **No reconstruction step.** The stim centre and radius are mutated
    directly into the tracks dataframe by ``RandomStimPerCellCircle``
    at acquisition time, so they're already in ``exp_data.parquet`` —
    just read them off.
  * **Control selection** = pick K random pixels inside the cell whose
    distance-to-boundary is at least ``radius`` (so the disk fits) and
    are at least ``min_separation`` from the stim centre (so the
    control disk doesn't overlap the stim disk).
  * **Intensity reading** uses the disk mask
    ``(yy-cy)^2 + (xx-cx)^2 <= radius^2``, optionally AND-ed with the
    per-frame cell mask so background pixels don't poison the readout.
  * **DINO features** come from JAFAR-upscaled tokens whose centre lies
    inside the disk (averaged) — exactly the same idea as the patch
    version, just with a circular instead of square footprint.

Everything downstream of feature extraction (PCA, regressor, clustering,
trajectory heatmap, baseline normalisation) is identical to the patch
analysis — those helpers operate on the long-form features dataframe
which has the same shape regardless of stim geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# Eligible-centroid selection (mirrors RandomStimPerCellCircle._select_centroid)
# ----------------------------------------------------------------------------


def candidate_centroids_for_cell(
    labels_frame: np.ndarray, cell_id: int, radius: float
) -> np.ndarray:
    """Pixels inside ``cell_id`` where a disk of ``radius`` fits entirely.

    Returns an (N, 2) array of (y, x) pixel coordinates — every pixel
    whose Euclidean distance to the cell boundary is ≥ ``radius``.
    Mirrors :class:`RandomStimPerCellCircle._select_centroid_per_cell`.
    """
    mask = labels_frame == cell_id
    if not mask.any():
        return np.empty((0, 2), dtype=int)
    edt = distance_transform_edt(mask)
    ys, xs = np.where(edt >= radius)
    return np.stack([ys, xs], axis=1)


def select_control_patches(
    stim_table: pd.DataFrame,
    labels_loader,
    *,
    fov_col: str = "fov",
    particle_col: str = "particle",
    t_col: str = "fov_timestep",
    stim_cy_col: str = "stim_center_y",
    stim_cx_col: str = "stim_center_x",
    radius_col: str = "stim_radius",
    cell_id_col: str = "label",
    n_controls: int = 8,
    min_separation: float | None = None,
    seed_offset: int = 7,
) -> pd.DataFrame:
    """Pick K random control disk centres per cell.

    Eligibility = pixels inside the cell where a disk of ``stim_radius``
    fits entirely (``edt >= radius``). Candidates within
    ``min_separation`` pixels of the stim centre are excluded so the
    control disk doesn't overlap the stim disk; default is ``2*radius``
    (centres at least 2r apart → disks just touching).

    Use ``cell_id_col="label"`` and a per-frame **seg-labels** loader —
    not particles — so a ``particle=0`` cell isn't confused with
    background.

    Returns one row per ``(fov, particle, ctrl_idx)`` with columns
    ``ctrl_center_y``, ``ctrl_center_x``, ``ctrl_radius``. Cells with
    no eligible candidates outside the separation zone are dropped.
    """
    records: list[dict] = []
    for _, row in stim_table.iterrows():
        fov = int(row[fov_col])
        particle = int(row[particle_col])
        cell_id = int(row[cell_id_col])
        t = int(row[t_col])
        scy = float(row[stim_cy_col])
        scx = float(row[stim_cx_col])
        r = float(row[radius_col])
        sep = 2.0 * r if min_separation is None else float(min_separation)

        labels = labels_loader(t, fov)
        cands = candidate_centroids_for_cell(labels, cell_id, r)
        if len(cands) == 0:
            continue
        # Exclude candidates too close to the stim centre.
        dy = cands[:, 0] - scy
        dx = cands[:, 1] - scx
        keep = (dy * dy + dx * dx) > sep * sep
        cands = cands[keep]
        if len(cands) == 0:
            continue
        n_pick = min(int(n_controls), len(cands))
        seed = (fov * 1_000_003 + particle * 100_003 + seed_offset) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(cands), size=n_pick, replace=False)
        for k, idx in enumerate(picks):
            cy, cx = cands[int(idx)]
            records.append(
                {
                    fov_col: fov,
                    particle_col: particle,
                    "ctrl_idx": int(k),
                    "ctrl_center_y": float(cy),
                    "ctrl_center_x": float(cx),
                    "ctrl_radius": r,
                }
            )
    return pd.DataFrame(records)


def build_patches_long(
    df_exp: pd.DataFrame,
    ctrl_table: pd.DataFrame,
    *,
    fov_col: str = "fov",
    particle_col: str = "particle",
) -> pd.DataFrame:
    """Tall form: one row per (fov, particle, role) with the disk centre + radius.

    Columns: fov, particle, role ∈ {"stim", "control"}, center_y,
    center_x, radius. The stim half is read straight off ``df_exp``'s
    ``stim_center_y/x/radius`` columns (already mutated in by
    ``RandomStimPerCellCircle`` at acquisition time); the control half
    comes from :func:`select_control_patches`.
    """
    stim_src = (
        df_exp.dropna(subset=["stim_center_y", "stim_center_x"])
        .groupby([fov_col, particle_col])[["stim_center_y", "stim_center_x", "stim_radius"]]
        .first()
        .reset_index()
        .rename(
            columns={
                "stim_center_y": "center_y",
                "stim_center_x": "center_x",
                "stim_radius": "radius",
            }
        )
    )
    stim_src["role"] = "stim"

    ctrl_src = ctrl_table[
        [fov_col, particle_col, "ctrl_center_y", "ctrl_center_x", "ctrl_radius"]
    ].rename(
        columns={
            "ctrl_center_y": "center_y",
            "ctrl_center_x": "center_x",
            "ctrl_radius": "radius",
        }
    )
    ctrl_src["role"] = "control"

    out = pd.concat([stim_src, ctrl_src], ignore_index=True)
    out["center_y"] = out["center_y"].astype(float)
    out["center_x"] = out["center_x"].astype(float)
    out["radius"] = out["radius"].astype(float)
    return out


# ----------------------------------------------------------------------------
# Cell-mask loader (identical to patch version)
# ----------------------------------------------------------------------------


def make_cell_mask_loader(
    df_exp: pd.DataFrame,
    labels_seg_loader,
    *,
    fov_col: str = "fov",
    t_col: str = "fov_timestep",
    particle_col: str = "particle",
    label_col: str = "label",
):
    """``loader(t, fov, particle) -> (H, W) bool`` of the cell's pixels.

    Resolves ``(fov, fov_timestep, particle) -> per-frame seg label`` once
    from ``df_exp``, then at call time returns ``labels == seg_label``.
    Cells not tracked at a frame return an empty mask.
    """
    sub = (
        df_exp[[fov_col, t_col, particle_col, label_col]]
        .dropna(subset=[particle_col])
    )
    lookup: dict[tuple[int, int, int], int] = dict(
        zip(
            zip(
                sub[fov_col].astype(int),
                sub[t_col].astype(int),
                sub[particle_col].astype(int),
            ),
            sub[label_col].astype(int),
        )
    )

    def loader(t: int, fov: int, particle: int) -> np.ndarray:
        labels = labels_seg_loader(int(t), int(fov))
        seg_label = lookup.get((int(fov), int(t), int(particle)))
        if seg_label is None:
            return np.zeros(labels.shape, dtype=bool)
        return labels == seg_label

    return loader


# ----------------------------------------------------------------------------
# Feature extractor protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractor(Protocol):
    """Adds columns to the long extracted-features table.

    Lifecycle per (fov, t) frame:
      1. driver loads ``frame_imgs`` ((C, H, W)) from zarr once
      2. driver calls ``prepare_frame(frame_imgs, t, fov)`` on each extractor
      3. driver calls ``extract(..., cell_mask=...)`` once per patch

    For the circle experiment, ``extract`` takes ``(center_y, center_x, radius)``
    in pixel coordinates rather than a bounding box.
    """

    @property
    def column_names(self) -> tuple[str, ...]: ...

    def prepare_frame(
        self, frame_imgs: np.ndarray, t: int, fov: int
    ) -> None: ...

    def extract(
        self,
        frame_imgs: np.ndarray,
        center_y: float,
        center_x: float,
        radius: float,
        cell_mask: np.ndarray | None = None,
    ) -> dict[str, float]: ...


def _noop_prepare_frame(self, frame_imgs, t, fov):
    return None


def _disk_bbox(center_y: float, center_x: float, radius: float, H: int, W: int):
    """Bounding-box of a disk clipped to image bounds."""
    r_int = int(np.ceil(radius))
    y0 = max(0, int(np.floor(center_y)) - r_int)
    y1 = min(H, int(np.ceil(center_y)) + r_int + 1)
    x0 = max(0, int(np.floor(center_x)) - r_int)
    x1 = min(W, int(np.ceil(center_x)) + r_int + 1)
    return y0, y1, x0, x1


def _disk_mask(
    center_y: float, center_x: float, radius: float, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """Boolean disk mask sliced to ``[y0:y1, x0:x1]``."""
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2


# ----------------------------------------------------------------------------
# MeanIntensity — disk readout, optionally masked to cell pixels
# ----------------------------------------------------------------------------


@dataclass
class MeanIntensity:
    """Mean pixel intensity inside the measurement disk per channel.

    The optogenetic effect spreads beyond the small (~3 px) stim disk, so
    the readout should be measured over a LARGER region. Set
    ``measure_radius`` to the desired measurement radius; the readout
    disk is then ``measure_radius`` centred on the same stim/control
    centre, NOT the ~3 px stim disk. ``measure_radius=None`` falls back
    to the per-patch radius (the stim disk itself).

    The disk is always intersected with the per-frame cell mask (when
    ``cell_mask`` is given) so background pixels are never counted — at a
    20 px radius the disk routinely overruns the cell boundary, and
    counting background there would manufacture spurious response.

    Emits ``<prefix>dot_npix``: how many cell pixels actually fell inside
    the disk on that frame. Downstream QC uses it to drop trajectories
    whose cell mask drifts off the measurement disk (overlap collapses to
    a handful of pixels — unreliable intensity).

    Column names keep the ``dot_mean_<ch>`` pattern so the
    baseline-normalisation / plotting code works unchanged.
    """

    channels: dict[str, int]
    prefix: str = ""
    measure_radius: float | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        cols = [f"{self.prefix}dot_mean_{ch}" for ch in self.channels]
        cols.append(f"{self.prefix}dot_npix")
        return tuple(cols)

    prepare_frame = _noop_prepare_frame

    def extract(self, frame_imgs, center_y, center_x, radius, cell_mask=None):
        r = self.measure_radius if self.measure_radius is not None else radius
        H, W = frame_imgs.shape[-2:]
        y0, y1, x0, x1 = _disk_bbox(center_y, center_x, r, H, W)
        npix_col = f"{self.prefix}dot_npix"
        if y0 >= y1 or x0 >= x1:
            return {c: np.nan for c in self.column_names}
        disk = _disk_mask(center_y, center_x, r, y0, y1, x0, x1)
        if cell_mask is not None:
            disk = disk & cell_mask[y0:y1, x0:x1]
        npix = int(disk.sum())
        out: dict[str, float] = {npix_col: float(npix)}
        if npix == 0:
            for ch in self.channels:
                out[f"{self.prefix}dot_mean_{ch}"] = np.nan
            return out
        for ch, ci in self.channels.items():
            tile = frame_imgs[ci, y0:y1, x0:x1]
            out[f"{self.prefix}dot_mean_{ch}"] = float(tile[disk].mean())
        return out


# ----------------------------------------------------------------------------
# RadialProfile — cumulative-disk means at a range of radii
# ----------------------------------------------------------------------------


@dataclass
class RadialProfile:
    """Cumulative-disk mean intensity at a range of radii.

    For every radius in ``radii``, emits the cell-masked mean intensity of
    the disk of that radius around the stim/control centre, plus the
    cell-pixel count. The radial-profile diagnostic baseline-normalises
    these and plots response-fold vs radius: the optogenetic response is
    concentrated near the spot and dilutes as the disk widens, so the
    radius that maximises the stim-vs-control contrast is the one to use
    for ``MEASURE_RADIUS``.

    This is a diagnostic extractor — run it via ``extract_features`` on a
    small SAMPLE of patches (no need to profile every cell).
    """

    channels: dict[str, int]
    radii: tuple[float, ...] = (3, 5, 8, 12, 16, 20, 26, 32)
    prefix: str = "prof_"

    @property
    def column_names(self) -> tuple[str, ...]:
        cols: list[str] = []
        for r in self.radii:
            ri = int(round(r))
            for ch in self.channels:
                cols.append(f"{self.prefix}{ch}_r{ri}")
            cols.append(f"{self.prefix}npix_r{ri}")
        return tuple(cols)

    prepare_frame = _noop_prepare_frame

    def extract(self, frame_imgs, center_y, center_x, radius, cell_mask=None):
        H, W = frame_imgs.shape[-2:]
        out = {c: np.nan for c in self.column_names}
        r_max = float(self.radii[-1])
        y0, y1, x0, x1 = _disk_bbox(center_y, center_x, r_max, H, W)
        if y0 >= y1 or x0 >= x1:
            return out
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
        cm = (
            cell_mask[y0:y1, x0:x1]
            if cell_mask is not None
            else np.ones((y1 - y0, x1 - x0), dtype=bool)
        )
        tiles = {ch: frame_imgs[ci, y0:y1, x0:x1] for ch, ci in self.channels.items()}
        for r in self.radii:
            ri = int(round(r))
            disk = (dist2 <= r * r) & cm
            npix = int(disk.sum())
            out[f"{self.prefix}npix_r{ri}"] = float(npix)
            if npix > 0:
                for ch, tile in tiles.items():
                    out[f"{self.prefix}{ch}_r{ri}"] = float(tile[disk].mean())
        return out


# ----------------------------------------------------------------------------
# GLCMTexture — window centred on the disk centre
# ----------------------------------------------------------------------------


@dataclass
class GLCMTexture:
    """Texture features on a square window centred on the disk centre.

    Window context isn't masked to the cell — texture is meant to
    describe local *image* structure around the spot, not pixels
    strictly inside the cell.
    """

    channels: dict[str, int]
    window_size: int = 52
    glcm_levels: int = 32
    prefix: str = "tex_"

    _FEATS = (
        "mean",
        "std",
        "p5_p95_range",
        "sobel_mean",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
    )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.prefix}{ch}_{f}" for ch in self.channels for f in self._FEATS
        )

    prepare_frame = _noop_prepare_frame

    def extract(self, frame_imgs, center_y, center_x, radius, cell_mask=None):
        from skimage.feature import graycomatrix, graycoprops
        from skimage.filters import sobel

        H, W = frame_imgs.shape[-2:]
        half = self.window_size // 2
        y0 = int(round(center_y - half))
        x0 = int(round(center_x - half))
        y1 = y0 + self.window_size
        x1 = x0 + self.window_size
        yy0, xx0 = max(0, y0), max(0, x0)
        yy1, xx1 = min(H, y1), min(W, x1)

        out: dict[str, float] = {}
        for ch, ci in self.channels.items():
            win = np.zeros(
                (self.window_size, self.window_size), dtype=frame_imgs.dtype
            )
            if yy0 < yy1 and xx0 < xx1:
                src = frame_imgs[ci, yy0:yy1, xx0:xx1]
                win[
                    yy0 - y0 : yy0 - y0 + src.shape[0],
                    xx0 - x0 : xx0 - x0 + src.shape[1],
                ] = src
            out.update(self._features(win, ch, graycomatrix, graycoprops, sobel))
        return out

    def _features(self, win, ch, graycomatrix, graycoprops, sobel):
        p = self.prefix
        if win.size == 0 or win.max() == 0:
            return {f"{p}{ch}_{f}": np.nan for f in self._FEATS}
        w = win.astype(np.float32)
        lo, hi = w.min(), w.max()
        q = (
            ((w - lo) / (hi - lo) * (self.glcm_levels - 1)).astype(np.uint8)
            if hi > lo
            else np.zeros_like(w, dtype=np.uint8)
        )
        glcm = graycomatrix(
            q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=self.glcm_levels,
            symmetric=True,
            normed=True,
        )
        p5, p95 = np.percentile(w, [5, 95])
        return {
            f"{p}{ch}_mean": float(w.mean()),
            f"{p}{ch}_std": float(w.std()),
            f"{p}{ch}_p5_p95_range": float(p95 - p5),
            f"{p}{ch}_sobel_mean": float(sobel(w / max(1.0, w.max())).mean()),
            f"{p}{ch}_glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
            f"{p}{ch}_glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            f"{p}{ch}_glcm_energy": float(graycoprops(glcm, "energy").mean()),
            f"{p}{ch}_glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        }


# ----------------------------------------------------------------------------
# DINOv3 + JAFAR upscaler — same as patch version, with disk-area pooling
# ----------------------------------------------------------------------------


@dataclass
class DINOv3JafarPatch:
    """Upscaled ViT features via JAFAR, averaged over tokens in the disk.

    Same backbone + JAFAR head as the patch analysis (see experiment 23's
    docstring for license / weights notes). The only change is that
    ``extract`` averages the upscaled tokens whose grid centres fall
    inside the stim disk, instead of inside the 14 px square patch.
    """

    channel_idx: int
    backbone_name: str = "vit_small_plus_patch16_dinov3.lvd1689m"
    jafar_weight_url: str = (
        "https://github.com/PaulCouairon/JAFAR/releases/download/Weights/"
        "vit_small_plus_patch16_dinov3.lvd1689m.pth"
    )
    jafar_weight_file: str = "vit_small_plus_patch16_dinov3.lvd1689m.pth"
    vit_patch_size: int = 16
    output_stride: int = 4
    feat_dim: int = 384
    prefix: str = "dino_"
    device: str = "cuda"
    _imagenet_mean = (0.485, 0.456, 0.406)
    _imagenet_std = (0.229, 0.224, 0.225)

    _backbone: object = field(default=None, init=False, repr=False)
    _jafar: object = field(default=None, init=False, repr=False)
    _device: object = field(default=None, init=False, repr=False)
    _features: np.ndarray | None = field(default=None, init=False, repr=False)
    _frame_shape: tuple[int, int] | None = field(default=None, init=False, repr=False)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}f{i:03d}" for i in range(self.feat_dim))

    def _load(self):
        if self._backbone is not None:
            return
        import os
        import torch
        from napari_convpaint.jafar.layers import PretrainedViTWrapper, JAFAR

        dev_name = self.device if torch.cuda.is_available() else "cpu"
        self._device = torch.device(dev_name)
        self._backbone = (
            PretrainedViTWrapper(name=self.backbone_name).eval().to(self._device)
        )

        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        local = os.path.join(cache_dir, self.jafar_weight_file)
        if not os.path.exists(local):
            torch.hub.download_url_to_file(self.jafar_weight_url, local)
        state = torch.load(local, map_location="cpu", weights_only=False)
        head = JAFAR(
            input_dim=3,
            qk_dim=128,
            v_dim=self.feat_dim,
            feature_dim=self.feat_dim,
            kernel_size=1,
            num_heads=4,
        )
        head.load_state_dict(state.get("jafar", state))
        self._jafar = head.eval().to(self._device)

    def _preprocess(self, img2d: np.ndarray):
        x = img2d.astype(np.float32)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        x = np.stack([x, x, x], axis=0)
        mean = np.array(self._imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self._imagenet_std, dtype=np.float32).reshape(3, 1, 1)
        return (x - mean) / std

    def prepare_frame(self, frame_imgs: np.ndarray, t: int, fov: int) -> None:
        import torch

        self._load()
        img = np.asarray(frame_imgs[self.channel_idx])
        H, W = int(img.shape[-2]), int(img.shape[-1])
        align = max(self.vit_patch_size, self.output_stride)
        pad_h = (-H) % align
        pad_w = (-W) % align
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        Hp, Wp = img.shape

        x = self._preprocess(img)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self._device)
        out_h = Hp // self.output_stride
        out_w = Wp // self.output_stride
        with torch.no_grad():
            lr_feats, _ = self._backbone(tensor)
            hr_feats = self._jafar(tensor, lr_feats, (out_h, out_w))
        grid = hr_feats[0].permute(1, 2, 0).cpu().numpy()
        n_h_orig = H // self.output_stride or 1
        n_w_orig = W // self.output_stride or 1
        if (n_h_orig, n_w_orig) != grid.shape[:2]:
            grid = grid[:n_h_orig, :n_w_orig]
        self._features = grid
        self._frame_shape = (H, W)

    def extract(self, frame_imgs, center_y, center_x, radius, cell_mask=None):
        if self._features is None or self._frame_shape is None:
            return {c: np.nan for c in self.column_names}
        n_h, n_w = self._features.shape[:2]
        s = self.output_stride
        # Token-grid bbox covering the disk's pixel extent.
        gi_lo = max(0, int((center_y - radius) // s))
        gi_hi = min(n_h, int(np.ceil((center_y + radius) / s)))
        gj_lo = max(0, int((center_x - radius) // s))
        gj_hi = min(n_w, int(np.ceil((center_x + radius) / s)))
        if gi_lo >= gi_hi or gj_lo >= gj_hi:
            return {c: np.nan for c in self.column_names}

        # Token centres (in pixel coords) within the bbox.
        gi = np.arange(gi_lo, gi_hi)
        gj = np.arange(gj_lo, gj_hi)
        tok_y = (gi + 0.5) * s
        tok_x = (gj + 0.5) * s
        yy, xx = np.meshgrid(tok_y, tok_x, indexing="ij")
        in_disk = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2

        if not in_disk.any():
            # Disk smaller than one token — fall back to the nearest token.
            gi0 = max(0, min(n_h - 1, int(center_y // s)))
            gj0 = max(0, min(n_w - 1, int(center_x // s)))
            vec = np.asarray(self._features[gi0, gj0]).reshape(-1)
        else:
            block = self._features[gi_lo:gi_hi, gj_lo:gj_hi]
            vec = block[in_disk].mean(axis=0)
        return {f"{self.prefix}f{i:03d}": float(v) for i, v in enumerate(vec)}


# ----------------------------------------------------------------------------
# Frame-major extraction driver
# ----------------------------------------------------------------------------


def extract_features(
    raw_zarr,
    patches_long: pd.DataFrame,
    extractors: list[FeatureExtractor],
    *,
    timepoints: Iterable[int] | None = None,
    n_imaging_channels: int | None = None,
    progress_every: int = 2000,
    cell_mask_loader=None,
) -> pd.DataFrame:
    """One pass over (fov, t). Each (t, fov) frame is loaded from zarr once.

    For each patch on that frame: optionally load the cell mask and pass
    it together with ``(center_y, center_x, radius)`` to every extractor.

    Aggregates duplicate ``(fov, particle, role, fov_timestep)`` rows by
    averaging — so ``n_controls > 1`` collapses naturally to one row per
    cell × role × frame.
    """
    T, P, C, H, W = raw_zarr.shape
    if timepoints is None:
        timepoints = list(range(T))
    else:
        timepoints = list(timepoints)
    if n_imaging_channels is None:
        n_imaging_channels = C

    by_fov: dict[int, pd.DataFrame] = {
        int(f): g.reset_index(drop=True) for f, g in patches_long.groupby("fov")
    }

    out_rows: list[dict] = []
    total_frames = sum(len(by_fov[fov]) and len(timepoints) for fov in by_fov)
    pbar = tqdm(
        total=total_frames, unit="frame", desc="extracting",
        disable=progress_every == 0,
    )
    for fov in sorted(by_fov):
        patches = by_fov[fov]
        for t in timepoints:
            frame = np.asarray(raw_zarr[int(t), int(fov), :n_imaging_channels])
            for ext in extractors:
                prep = getattr(ext, "prepare_frame", None)
                if prep is not None:
                    prep(frame, int(t), int(fov))
            for _, p in patches.iterrows():
                rec: dict[str, float | int | str] = {
                    "fov": fov,
                    "particle": int(p["particle"]),
                    "role": p["role"],
                    "fov_timestep": int(t),
                }
                cy = float(p["center_y"])
                cx = float(p["center_x"])
                r = float(p["radius"])
                cm = None
                if cell_mask_loader is not None:
                    cm = cell_mask_loader(int(t), int(fov), int(p["particle"]))
                for ext in extractors:
                    rec.update(ext.extract(frame, cy, cx, r, cell_mask=cm))
                out_rows.append(rec)
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(out_rows)
    keys = ["fov", "particle", "role", "fov_timestep"]
    feat_cols = [c for c in df.columns if c not in keys]
    if feat_cols:
        df = df.groupby(keys, as_index=False)[feat_cols].mean()
    return df


# ----------------------------------------------------------------------------
# Reshape / baseline-normalize (verbatim from patch analysis)
# ----------------------------------------------------------------------------


def pivot_roles(features_long: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [
        c
        for c in features_long.columns
        if c not in ("fov", "particle", "role", "fov_timestep")
    ]
    wide = features_long.pivot_table(
        index=["fov", "particle", "fov_timestep"],
        columns="role",
        values=feat_cols,
        aggfunc="first",
    )
    wide.columns = [f"{feat}_{role}" for feat, role in wide.columns]
    return wide.reset_index()


def baseline_normalize(
    features_long: pd.DataFrame,
    baseline_frames: Iterable[int],
    columns: Iterable[str],
    *,
    method: str = "median",
) -> pd.DataFrame:
    baseline_frames = list(baseline_frames)
    columns = list(columns)
    if not baseline_frames or not columns:
        return features_long
    stale = [
        f"{c}{suffix}"
        for c in columns
        for suffix in ("_baseline", "_fold")
        if f"{c}{suffix}" in features_long.columns
    ]
    if stale:
        features_long = features_long.drop(columns=stale)
    mask = features_long["fov_timestep"].isin(baseline_frames)
    agg = "median" if method == "median" else "mean"
    base = (
        features_long[mask]
        .groupby(["fov", "particle", "role"])[columns]
        .agg(agg)
        .add_suffix("_baseline")
        .reset_index()
    )
    out = features_long.merge(base, on=["fov", "particle", "role"], how="left")
    for c in columns:
        out[f"{c}_fold"] = out[c] / out[f"{c}_baseline"]
    return out


# ----------------------------------------------------------------------------
# Preview: draw stim & control disks (and texture window) on sample cells
# ----------------------------------------------------------------------------


def plot_patch_overlays(
    raw_zarr,
    labels_perframe,
    patches_long: pd.DataFrame,
    t: int,
    *,
    n_cells: int = 6,
    channel_idx: int = 0,
    texture_window: int | None = None,
    crop: int = 200,
    rng_seed: int | None = None,
    cells: pd.DataFrame | None = None,
):
    """Stim + control disks on a random sample of cells at time ``t``."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pivot = patches_long.pivot_table(
        index=["fov", "particle"],
        columns="role",
        values=["center_y", "center_x", "radius"],
        aggfunc="first",
    ).dropna()
    if cells is not None:
        keys = pd.MultiIndex.from_frame(cells[["fov", "particle"]])
        pivot = pivot.loc[pivot.index.intersection(keys)]
    if len(pivot) == 0:
        raise RuntimeError("no (fov, particle) has both stim and control coords")
    sample = pivot.sample(
        min(n_cells, len(pivot)), random_state=rng_seed
    ).reset_index()

    H, W = raw_zarr.shape[-2:]
    half = crop // 2
    fig, axs = plt.subplots(
        1, len(sample), figsize=(2.8 * len(sample), 2.8), squeeze=False
    )
    for ax, (_, row) in zip(axs[0], sample.iterrows()):
        fov = int(row[("fov", "")])
        particle = int(row[("particle", "")])
        scy = float(row[("center_y", "stim")])
        scx = float(row[("center_x", "stim")])
        sr = float(row[("radius", "stim")])
        ccy = float(row[("center_y", "control")])
        ccx = float(row[("center_x", "control")])
        cr = float(row[("radius", "control")])

        y0 = max(0, int(scy) - half)
        y1 = min(H, y0 + crop)
        y0 = max(0, y1 - crop)
        x0 = max(0, int(scx) - half)
        x1 = min(W, x0 + crop)
        x0 = max(0, x1 - crop)

        img = np.asarray(raw_zarr[t, fov, channel_idx, y0:y1, x0:x1])
        lbl = np.asarray(labels_perframe[t, fov, y0:y1, x0:x1])
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.contour(lbl == particle, levels=[0.5], colors="lime", linewidths=0.6)

        ax.add_patch(
            mpatches.Circle(
                (scx - x0, scy - y0), sr,
                edgecolor="red", facecolor="none", linewidth=1.0,
            )
        )
        ax.add_patch(
            mpatches.Circle(
                (ccx - x0, ccy - y0), cr,
                edgecolor="cyan", facecolor="none", linewidth=1.0,
            )
        )
        if texture_window:
            tx = scx - x0 - texture_window / 2
            ty = scy - y0 - texture_window / 2
            ax.add_patch(
                mpatches.Rectangle(
                    (tx, ty),
                    texture_window, texture_window,
                    edgecolor="magenta", facecolor="none",
                    linewidth=0.6, linestyle="--",
                )
            )

        ax.set_title(f"fov={fov} particle={particle}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"t={t}  red=stim disk  cyan=control disk"
        + ("  magenta=texture" if texture_window else "")
    )
    fig.tight_layout()
    return fig, axs


# ----------------------------------------------------------------------------
# DINO PCA preview (per-cell + global) — verbatim from patch analysis
# ----------------------------------------------------------------------------


def dino_features_pca_rgb_per_cell(
    features_grid: np.ndarray,
    cell_labels: np.ndarray,
    *,
    n_components: int = 3,
    min_tokens_per_cell: int = 6,
    robust: bool = True,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Per-cell PCA → (n_h, n_w, 3) RGB. Each cell gets its own basis so
    the colours describe within-cell variation, not positional artefacts."""
    from sklearn.decomposition import PCA

    if features_grid.shape[:2] != cell_labels.shape:
        raise ValueError(
            f"shape mismatch: features_grid {features_grid.shape[:2]} vs "
            f"cell_labels {cell_labels.shape}"
        )
    n_h, n_w, d = features_grid.shape
    rgb = np.empty((n_h, n_w, 3), dtype=np.float32)
    rgb[:] = np.asarray(background_color, dtype=np.float32)
    for label in np.unique(cell_labels):
        if label == 0:
            continue
        mask = cell_labels == label
        n_tokens = int(mask.sum())
        if n_tokens < min_tokens_per_cell:
            continue
        toks = features_grid[mask]
        pca = PCA(n_components=n_components, random_state=0).fit(toks)
        pcs = pca.transform(toks)
        for c in range(n_components):
            ch = pcs[:, c]
            if robust and n_tokens >= 10:
                lo, hi = np.percentile(ch, [2, 98])
            else:
                lo, hi = ch.min(), ch.max()
            pcs[:, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        rgb[mask] = pcs.astype(np.float32)
    return rgb


def dino_features_pca_rgb(
    features_grid: np.ndarray,
    *,
    pca: object | None = None,
    robust: bool = True,
) -> tuple[np.ndarray, object]:
    """Global PCA → (n_h, n_w, 3) RGB. Kept as a baseline; per-cell is
    usually more informative for subcellular structure."""
    from sklearn.decomposition import PCA

    n_h, n_w, d = features_grid.shape
    flat = features_grid.reshape(n_h * n_w, d)
    if pca is None:
        pca = PCA(n_components=3, random_state=0).fit(flat)
    pcs = pca.transform(flat).reshape(n_h, n_w, 3)
    rgb = np.empty_like(pcs)
    for c in range(3):
        ch = pcs[..., c]
        if robust:
            lo, hi = np.percentile(ch, [2, 98])
        else:
            lo, hi = ch.min(), ch.max()
        rgb[..., c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb, pca


# ----------------------------------------------------------------------------
# Responsiveness analysis (verbatim from patch analysis)
# ----------------------------------------------------------------------------


def per_cell_response_metric(
    features_long: pd.DataFrame,
    *,
    column: str,
    response_frames: Iterable[int],
    agg: str = "max",
    role: str = "stim",
    fov_col: str = "fov",
    particle_col: str = "particle",
) -> pd.DataFrame:
    """Per-cell scalar response, aggregated over ``response_frames``.

    ``response_frames`` is the window the response is *measured* over —
    NOT the stim frames. The optogenetic actuator (FGFR) is upstream of
    the readout (PIP), so the downstream signal lags the stim block by
    several frames and the peak fold-change lands well after stimulation
    ends. Pass a window that runs from stim onset to the end of the movie
    (or at least far enough past the stim block to catch the peak), with
    ``agg="max"`` so the delayed peak is captured wherever it occurs.
    """
    response_frames = list(response_frames)
    mask = (features_long["role"] == role) & (
        features_long["fov_timestep"].isin(response_frames)
    )
    return (
        features_long.loc[mask]
        .groupby([fov_col, particle_col])[column]
        .agg(agg)
        .rename("response")
        .reset_index()
    )


def per_cell_dino_vector(
    features_long: pd.DataFrame,
    *,
    frames: Iterable[int],
    role: str = "stim",
    dino_prefix: str = "dino_",
    fov_col: str = "fov",
    particle_col: str = "particle",
) -> tuple[np.ndarray, pd.DataFrame]:
    frames = list(frames)
    dino_cols = [c for c in features_long.columns if c.startswith(dino_prefix)]
    if not dino_cols:
        raise RuntimeError(
            f"no columns starting with {dino_prefix!r} in features_long; "
            "did you run the extraction with DINOv3JafarPatch enabled?"
        )
    sub = features_long[
        (features_long["role"] == role)
        & (features_long["fov_timestep"].isin(frames))
    ]
    agg = sub.groupby([fov_col, particle_col])[dino_cols].mean().reset_index()
    cells_df = agg[[fov_col, particle_col]].reset_index(drop=True)
    X = agg[dino_cols].to_numpy(dtype=np.float32)
    return X, cells_df


def fit_response_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int = 5,
    random_state: int = 0,
    standardize: bool = True,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X) if standardize else None
    Xs = scaler.transform(X) if scaler is not None else X
    rf = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
    lr = LogisticRegression(max_iter=2000, random_state=random_state)
    cv_auc_rf = cross_val_score(rf, Xs, y, cv=cv_folds, scoring="roc_auc").mean()
    cv_auc_lr = cross_val_score(lr, Xs, y, cv=cv_folds, scoring="roc_auc").mean()
    rf.fit(Xs, y)
    lr.fit(Xs, y)
    return {
        "rf": rf, "logreg": lr, "scaler": scaler,
        "cv_auc_rf": float(cv_auc_rf), "cv_auc_lr": float(cv_auc_lr),
        "importance_rf": np.asarray(rf.feature_importances_),
        "importance_lr": np.abs(lr.coef_[0]),
    }


def fit_response_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int = 5,
    random_state: int = 0,
    standardize: bool = True,
):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X) if standardize else None
    Xs = scaler.transform(X) if scaler is not None else X
    rf = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    ridge = Ridge(alpha=1.0, random_state=random_state)
    cv_r2_rf = cross_val_score(rf, Xs, y, cv=cv_folds, scoring="r2").mean()
    cv_r2_ridge = cross_val_score(ridge, Xs, y, cv=cv_folds, scoring="r2").mean()
    rf.fit(Xs, y)
    ridge.fit(Xs, y)
    return {
        "rf": rf, "ridge": ridge, "scaler": scaler,
        "cv_r2_rf": float(cv_r2_rf), "cv_r2_ridge": float(cv_r2_ridge),
        "importance_rf": np.asarray(rf.feature_importances_),
    }


def predict_response_map(
    model,
    features_grid: np.ndarray,
    *,
    scaler=None,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    n_h, n_w, d = features_grid.shape
    X = features_grid.reshape(n_h * n_w, d)
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X).reshape(n_h, n_w)
    if image_shape is not None:
        H, W = image_shape
        sy = max(1, H // n_h)
        sx = max(1, W // n_w)
        preds = np.repeat(np.repeat(preds, sy, axis=0), sx, axis=1)
        preds = preds[:H, :W]
    return preds


def cluster_dino_features(
    X: np.ndarray,
    *,
    n_clusters: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(X)


def dino_feature_map(
    features_grid: np.ndarray,
    feature_index: int,
    *,
    image_shape: tuple[int, int] | None = None,
    robust: bool = True,
) -> np.ndarray:
    n_h, n_w, _ = features_grid.shape
    m = np.asarray(features_grid[..., feature_index], dtype=np.float32)
    if robust:
        lo, hi = np.percentile(m, [2, 98])
    else:
        lo, hi = m.min(), m.max()
    m = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    if image_shape is not None:
        H, W = image_shape
        sy = max(1, H // n_h)
        sx = max(1, W // n_w)
        m = np.repeat(np.repeat(m, sy, axis=0), sx, axis=1)
        m = m[:H, :W]
    return m
