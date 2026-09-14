"""Helpers for per-cell stim-response analysis.

The notebook owns orchestration (loading, previewing, filtering, plotting).
This module owns the heavy lifting:

  * Picking a control patch inside each cell that mirrors the stimulator's
    eligibility rules but excludes the stim patch itself.
  * One pass over (fov, fov_timestep) that loads each frame's raw image from
    zarr exactly once and applies every registered `FeatureExtractor` to
    every patch on that frame.
  * Preview helpers so you can dial in patch / window sizes against the
    actual data *before* you commit to the expensive extraction.

Design choices:

  * Extraction is pure (no QC, no filtering). Re-cut your selection
    criteria in the notebook without re-reading any pixels.
  * The driver loops over frames in the outer loop and patches in the
    inner loop so the (T, P, C, H, W) zarr is only touched once per (t, p).
  * Adding a feature = add a class with `column_names` + `extract(...)`.
    The driver does not care what the features are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# Patch eligibility / control selection
# ----------------------------------------------------------------------------


def candidate_patches_for_cell(
    labels_frame: np.ndarray, cell_id: int, patch_size: int
) -> np.ndarray:
    """Patches fully inside ``cell_id`` in a (Y, X) label frame.

    Returns an (N, 2) array of (i, j) grid indices. Same eligibility rule as
    ``RandomStimPerCell14pxPatches._select_patch_per_cell``, restricted to a
    single cell id.
    """
    ps = patch_size
    h, w = labels_frame.shape
    n_h, n_w = h // ps, w // ps
    if n_h == 0 or n_w == 0:
        return np.empty((0, 2), dtype=int)
    cropped = labels_frame[: n_h * ps, : n_w * ps]
    blocks = cropped.reshape(n_h, ps, n_w, ps)
    block_min = blocks.min(axis=(1, 3))
    block_max = blocks.max(axis=(1, 3))
    uniform = (block_min == block_max) & (block_min == cell_id)
    ii, jj = np.nonzero(uniform)
    return np.stack([ii, jj], axis=1)


def select_control_patches(
    stim_table: pd.DataFrame,
    labels_loader,
    *,
    fov_col: str = "fov",
    particle_col: str = "particle",
    t_col: str = "fov_timestep",
    stim_i_col: str = "patch_i",
    stim_j_col: str = "patch_j",
    cell_id_col: str = "particle",
    n_controls: int = 1,
    patch_size: int = 14,
    seed_offset: int = 7,
) -> pd.DataFrame:
    """Pick one control patch per (fov, particle).

    The control patch is uniformly random among patches that fit fully
    inside the cell, excluding the stim patch. Seed is deterministic from
    (fov, particle) so the choice is reproducible.

    Parameters
    ----------
    stim_table
        One row per (fov, particle) that has a stim patch. Must carry the
        stim grid coords (``stim_i_col`` / ``stim_j_col``), a frame
        (``t_col``), and the column named by ``cell_id_col`` (used to
        find candidate patches inside this cell).
    labels_loader
        ``labels_loader(t, fov) -> np.ndarray`` returning the labels
        image whose pixel values correspond to ``cell_id_col``.
    cell_id_col
        Which column in ``stim_table`` holds the cell id used for
        eligibility lookups (must match the values in the labels image
        returned by ``labels_loader``). Default ``"particle"`` is
        WRONG for typical faro datasets — trackpy can legitimately
        assign ``particle=0`` to a real cell, but ``labels_to_particles``
        initializes the particle array to 0 for background, so a
        ``particle=0`` cell becomes indistinguishable from background.
        Pass ``cell_id_col="label"`` and a per-frame seg-labels loader
        for unambiguous results.
    n_controls
        How many control patches to pick per cell. Multiple controls per
        cell shrink the per-cell sampling variance: with K controls the
        per-cell control mean has ~1/√K the noise of a single pick.
        Cells with fewer than ``n_controls + 1`` eligible patches emit
        as many controls as they can support.

    Returns
    -------
    One row per (fov, particle, ctrl_idx) with ctrl_idx in
    ``[0, min(n_controls, n_candidates - 1))``. Cells with < 2 candidates
    in total are dropped (no room for both a stim and a control).
    Columns: fov, particle, ctrl_idx, ctrl_patch_i, ctrl_patch_j,
    ctrl_patch_y_min, ctrl_patch_x_min.
    """
    records: list[dict] = []
    for _, row in stim_table.iterrows():
        fov = int(row[fov_col])
        particle = int(row[particle_col])
        cell_id = int(row[cell_id_col])
        t = int(row[t_col])
        labels = labels_loader(t, fov)
        cands = candidate_patches_for_cell(labels, cell_id, patch_size)
        if len(cands) < 2:
            continue
        si = int(row[stim_i_col])
        sj = int(row[stim_j_col])
        cands = cands[~((cands[:, 0] == si) & (cands[:, 1] == sj))]
        if len(cands) == 0:
            continue
        n_pick = min(int(n_controls), len(cands))
        seed = (fov * 1_000_003 + particle * 100_003 + seed_offset) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(cands), size=n_pick, replace=False)
        for k, idx in enumerate(picks):
            pi, pj = cands[int(idx)]
            records.append(
                {
                    fov_col: fov,
                    particle_col: particle,
                    "ctrl_idx": int(k),
                    "ctrl_patch_i": int(pi),
                    "ctrl_patch_j": int(pj),
                    "ctrl_patch_y_min": int(pi) * patch_size,
                    "ctrl_patch_x_min": int(pj) * patch_size,
                }
            )
    return pd.DataFrame(records)


def make_cell_mask_loader(
    df_exp: pd.DataFrame,
    labels_seg_loader,
    *,
    fov_col: str = "fov",
    t_col: str = "fov_timestep",
    particle_col: str = "particle",
    label_col: str = "label",
):
    """Build a ``loader(t, fov, particle) -> (H, W) bool`` that returns the
    pixel mask of the tracked cell at that frame.

    Resolves ``(fov, fov_timestep, particle) -> per-frame seg label`` once
    from ``df_exp``, then at call time loads the seg-label image at that
    frame and returns ``labels == seg_label``. Cells not tracked at a
    frame return an empty mask so intensity extractors will NaN out
    rather than read background.

    Use seg labels (``labels/labels/0``), not particle labels — particle 0
    aliases with background in the particles image.
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
            # No seg label for this cell at this frame — emit an empty
            # mask so intensity extractors return NaN here.
            return np.zeros(labels.shape, dtype=bool)
        return labels == seg_label

    return loader


def build_patches_long(
    df_exp: pd.DataFrame,
    ctrl_table: pd.DataFrame,
    *,
    fov_col: str = "fov",
    particle_col: str = "particle",
) -> pd.DataFrame:
    """Tall form patches table: one row per (fov, particle, role).

    Columns: fov, particle, role ∈ {"stim", "control"}, y_min, x_min.

    Used to drive the frame-major feature extractor. Both stim and control
    rows share the same shape so extractors can treat them uniformly.
    """
    stim_src = (
        df_exp.dropna(subset=["patch_y_min", "patch_x_min"])
        .groupby([fov_col, particle_col])[["patch_y_min", "patch_x_min"]]
        .first()
        .reset_index()
        .rename(columns={"patch_y_min": "y_min", "patch_x_min": "x_min"})
    )
    stim_src["role"] = "stim"

    ctrl_src = ctrl_table[
        [fov_col, particle_col, "ctrl_patch_y_min", "ctrl_patch_x_min"]
    ].rename(columns={"ctrl_patch_y_min": "y_min", "ctrl_patch_x_min": "x_min"})
    ctrl_src["role"] = "control"

    out = pd.concat([stim_src, ctrl_src], ignore_index=True)
    out["y_min"] = out["y_min"].astype(int)
    out["x_min"] = out["x_min"].astype(int)
    return out


# ----------------------------------------------------------------------------
# Feature extractors
# ----------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractor(Protocol):
    """Adds columns to the long extracted-features table.

    Lifecycle per (fov, t) frame:
      1. driver loads ``frame_imgs`` (shape (C, H, W)) from zarr once
      2. driver calls ``prepare_frame(frame_imgs, t, fov)`` on each extractor
      3. driver calls ``extract(..., cell_mask=...)`` once per patch

    ``prepare_frame`` is where any expensive frame-wide work belongs (e.g.
    running a ViT once for the whole image and caching the patch features).
    ``extract`` should then be cheap — a lookup into whatever was cached.

    ``cell_mask`` (optional) is a per-frame boolean ``(H, W)`` mask of the
    pixels that belong to *this* cell. Extractors that read raw pixel
    intensities (``MeanIntensity``) should use it to ignore background so
    "response" doesn't get confounded by cells migrating into the stim
    region. Extractors that operate on global features (DINO) can ignore
    it.

    Implementations must:
      * Expose ``column_names`` — the columns they will contribute, in order.
      * Implement ``extract(frame_imgs, y_min, x_min, patch_size, cell_mask=None)``
        returning a dict keyed by exactly those column names.
      * Optionally override ``prepare_frame``. The default below is a no-op
        so cheap extractors don't need to care.
    """

    @property
    def column_names(self) -> tuple[str, ...]: ...

    def prepare_frame(
        self, frame_imgs: np.ndarray, t: int, fov: int
    ) -> None: ...

    def extract(
        self,
        frame_imgs: np.ndarray,
        y_min: int,
        x_min: int,
        patch_size: int,
        cell_mask: np.ndarray | None = None,
    ) -> dict[str, float]: ...


def _noop_prepare_frame(self, frame_imgs, t, fov):
    """Default per-frame hook for stateless extractors."""
    return None


def make_dot_mask(patch_size: int, dot_diameter: float) -> np.ndarray:
    center = (patch_size - 1) / 2
    r = dot_diameter / 2
    yy, xx = np.ogrid[:patch_size, :patch_size]
    return ((yy - center) ** 2 + (xx - center) ** 2) <= r**2


@dataclass
class MeanIntensity:
    """Per-channel mean over the full patch and optionally over a dot mask.

    The DMD only illuminates the dot, so `dot_mean` is the cleaner stim
    readout; `mean` is whole-patch (mixes lit and unlit pixels).
    """

    channels: dict[str, int]
    dot_mask: np.ndarray | None = None
    prefix: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        cols = [f"{self.prefix}mean_{ch}" for ch in self.channels]
        if self.dot_mask is not None:
            cols += [f"{self.prefix}dot_mean_{ch}" for ch in self.channels]
        return tuple(cols)

    prepare_frame = _noop_prepare_frame

    def extract(self, frame_imgs, y_min, x_min, patch_size, cell_mask=None):
        h, w = frame_imgs.shape[-2:]
        y1, x1 = y_min + patch_size, x_min + patch_size
        if y_min < 0 or x_min < 0 or y1 > h or x1 > w:
            return {c: np.nan for c in self.column_names}

        cell_tile = None
        if cell_mask is not None:
            cell_tile = cell_mask[y_min:y1, x_min:x1]
            # The "no cell pixels in this patch" case must NaN out every
            # channel — otherwise a cell that has drifted off the patch
            # would look like (small, background-noise) signal and bias
            # the response readout.
            if not cell_tile.any():
                return {c: np.nan for c in self.column_names}

        dot_mask = self.dot_mask
        dot_cell = None
        if dot_mask is not None and cell_tile is not None:
            dot_cell = dot_mask & cell_tile
            if not dot_cell.any():
                dot_cell = None  # disqualify the dot_mean for this row

        out: dict[str, float] = {}
        for ch, ci in self.channels.items():
            tile = frame_imgs[ci, y_min:y1, x_min:x1]
            if cell_tile is not None:
                out[f"{self.prefix}mean_{ch}"] = float(tile[cell_tile].mean())
            else:
                out[f"{self.prefix}mean_{ch}"] = float(tile.mean())
            if dot_mask is not None:
                if cell_tile is not None:
                    out[f"{self.prefix}dot_mean_{ch}"] = (
                        float(tile[dot_cell].mean()) if dot_cell is not None else np.nan
                    )
                else:
                    out[f"{self.prefix}dot_mean_{ch}"] = float(tile[dot_mask].mean())
        return out


@dataclass
class GLCMTexture:
    """Texture features on a square window centered on the patch.

    Reports mean / std / 5–95 IPR / mean Sobel / 4-direction GLCM
    (contrast, homogeneity, energy, correlation) per imaging channel.
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

    def extract(self, frame_imgs, y_min, x_min, patch_size, cell_mask=None):
        # GLCM needs a rectangular window — we don't mask to cell because
        # the texture is meant to describe the local *image* structure
        # around the patch, not just the pixels strictly inside the cell.
        from skimage.feature import graycomatrix, graycoprops
        from skimage.filters import sobel

        h, w = frame_imgs.shape[-2:]
        cy = y_min + (patch_size - 1) / 2
        cx = x_min + (patch_size - 1) / 2
        half = self.window_size // 2
        y0 = int(round(cy - half))
        x0 = int(round(cx - half))
        y1 = y0 + self.window_size
        x1 = x0 + self.window_size
        yy0, xx0 = max(0, y0), max(0, x0)
        yy1, xx1 = min(h, y1), min(w, x1)

        out: dict[str, float] = {}
        for ch, ci in self.channels.items():
            win = np.zeros((self.window_size, self.window_size), dtype=frame_imgs.dtype)
            if yy0 < yy1 and xx0 < xx1:
                src = frame_imgs[ci, yy0:yy1, xx0:xx1]
                win[yy0 - y0 : yy0 - y0 + src.shape[0], xx0 - x0 : xx0 - x0 + src.shape[1]] = src
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


@dataclass
class DINOv3Patch:
    """ViT patch-feature at the stim region (DINOv3 via timm).

    Runs DINOv3 ONCE per (fov, t) on the whole frame via ``prepare_frame``
    and caches the resulting (n_h, n_w, D) feature grid. ``extract`` is
    then just a grid-index lookup — for N patches on the same frame we
    pay one forward pass, not N.

    Why timm rather than Meta's ``torch.hub.load('facebookresearch/dinov3',…)``:
    timm mirrors the DINOv3 weights on HuggingFace Hub (e.g.
    ``timm/vit_small_patch16_dinov3.lvd1689m``) without Meta's license
    gate, so the first ``prepare_frame`` downloads without auth. The
    napari-convpaint dinov3 branch loads them the same way.

    Patch-size compatibility: DINOv2 ViT-S used a 14 px patch, which is
    why the current stim regions are 14 px (each region aligned 1:1 with
    one DINOv2 token). All DINOv3 ViT variants use 16 px, so the 14 px
    stim region doesn't tile evenly. The lookup picks the ViT token whose
    grid cell contains the stim centre; re-tune ``PATCH_SIZE`` to a
    multiple of ``vit_patch_size`` for exact alignment.

    Lazy load: timm is imported and the model is created+downloaded on
    first ``prepare_frame`` call, so the rest of the notebook runs even
    if you don't have GPU memory budget for the model.
    """

    channel_idx: int
    model_name: str = "vit_small_patch16_dinov3.lvd1689m"  # 384-D, 16-px patches, 5 prefix tokens
    vit_patch_size: int = 16
    feat_dim: int = 384
    prefix: str = "dino_"
    device: str = "cuda"
    # DINOv3 was trained with ImageNet-style stats; replicate the single
    # microscopy channel to 3 channels and apply.
    _imagenet_mean = (0.485, 0.456, 0.406)
    _imagenet_std = (0.229, 0.224, 0.225)

    _model: object = field(default=None, init=False, repr=False)
    _device: object = field(default=None, init=False, repr=False)
    _num_prefix_tokens: int = field(default=0, init=False, repr=False)
    _features: np.ndarray | None = field(default=None, init=False, repr=False)
    _frame_shape: tuple[int, int] | None = field(default=None, init=False, repr=False)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}f{i:03d}" for i in range(self.feat_dim))

    def _load(self):
        if self._model is not None:
            return
        import timm
        import torch

        dev_name = self.device if torch.cuda.is_available() else "cpu"
        self._device = torch.device(dev_name)
        # ``num_classes=0`` strips the classifier head so forward_features
        # returns the full token sequence (CLS + registers + patches).
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        model.eval().to(self._device)
        # DINOv3 has 1 CLS + 4 register tokens = 5 prefix tokens before
        # the spatial patch tokens. timm exposes the count.
        self._num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        self._model = model

    def _preprocess(self, img2d: np.ndarray):
        """Microscopy single-channel → 3ch ImageNet-normalised float32."""
        x = img2d.astype(np.float32)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        x = np.stack([x, x, x], axis=0)  # (3, H, W)
        mean = np.array(self._imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self._imagenet_std, dtype=np.float32).reshape(3, 1, 1)
        return (x - mean) / std

    def prepare_frame(self, frame_imgs: np.ndarray, t: int, fov: int) -> None:
        """Forward the whole frame through DINOv3, cache (n_h, n_w, D)."""
        import torch

        self._load()
        img = np.asarray(frame_imgs[self.channel_idx])
        H, W = int(img.shape[-2]), int(img.shape[-1])

        # ViT needs dims divisible by patch size. Reflect-pad to the next
        # multiple, then drop the padding-aligned tokens after the forward.
        pad_h = (-H) % self.vit_patch_size
        pad_w = (-W) % self.vit_patch_size
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        Hp, Wp = img.shape

        x = self._preprocess(img)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self._device)
        with torch.no_grad():
            tokens = self._model.forward_features(tensor)  # type: ignore[attr-defined]
        # tokens: (1, num_prefix + n_h*n_w, D). Strip the CLS+register
        # prefix; the remaining tokens are the spatial grid in row-major.
        patch_tokens = tokens[0, self._num_prefix_tokens :].cpu().numpy()
        n_h = Hp // self.vit_patch_size
        n_w = Wp // self.vit_patch_size
        grid = patch_tokens.reshape(n_h, n_w, -1)
        # Drop any tokens that lie entirely in the reflect-padded margin.
        n_h_orig = H // self.vit_patch_size or 1
        n_w_orig = W // self.vit_patch_size or 1
        if (n_h_orig, n_w_orig) != grid.shape[:2]:
            grid = grid[:n_h_orig, :n_w_orig]
        self._features = grid
        self._frame_shape = (H, W)

    def extract(self, frame_imgs, y_min, x_min, patch_size, cell_mask=None):
        # DINO features describe the whole patch globally — cell_mask is
        # not applied because masking would drop the surrounding context
        # the model uses to attend over.
        if self._features is None or self._frame_shape is None:
            return {c: np.nan for c in self.column_names}
        H, W = self._frame_shape
        n_h, n_w = self._features.shape[:2]
        scale_y = H / n_h
        scale_x = W / n_w
        cy = y_min + patch_size / 2
        cx = x_min + patch_size / 2
        gi = max(0, min(n_h - 1, int(cy / scale_y)))
        gj = max(0, min(n_w - 1, int(cx / scale_x)))
        vec = np.asarray(self._features[gi, gj]).reshape(-1)
        return {f"{self.prefix}f{i:03d}": float(v) for i, v in enumerate(vec)}


@dataclass
class DINOv3JafarPatch:
    """Upscaled ViT features via the JAFAR head (PaulCouairon/JAFAR).

    DINOv3 ViT-S+ produces patch features at (H/16, W/16) resolution. JAFAR
    is a small cross-attention upsampler trained on top of a frozen
    backbone that takes the input image (queries) and the LR ViT tokens
    (keys/values) and emits a higher-resolution feature map -- by default
    at (H/4, W/4) so each output token covers a 4x4 px region (16x finer
    than the raw ViT grid).

    Wiring mirrors napari-convpaint's ``DinoJafarFeatures`` (the dinov3
    branch): we instantiate ``PretrainedViTWrapper`` for the backbone and
    ``JAFAR`` for the head, both imported from
    ``napari_convpaint.jafar.layers``. The JAFAR weights live on the
    JAFAR project's GitHub releases page and are downloaded once into
    ``~/.cache/torch/hub/checkpoints/``.

    Per-patch feature: instead of picking the single ViT token closest to
    the stim centre (like :class:`DINOv3Patch` does), we average ALL
    upscaled tokens that overlap the patch's pixel footprint. With a 14 px
    stim patch and ``output_stride=4`` that's ~3x3 -- 4x4 tokens per
    patch, giving a feature that genuinely localises to the stim region
    rather than to the surrounding ViT token.

    Backbone is ``vit_small_plus_patch16_dinov3.lvd1689m`` (the variant
    the public JAFAR weights were trained against).
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

        self._backbone = PretrainedViTWrapper(name=self.backbone_name).eval().to(self._device)

        # JAFAR head weights are not on HF — pull from the JAFAR github
        # releases page once and cache locally.
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        local = os.path.join(cache_dir, self.jafar_weight_file)
        if not os.path.exists(local):
            torch.hub.download_url_to_file(self.jafar_weight_url, local)
        state = torch.load(local, map_location="cpu", weights_only=False)
        head = JAFAR(
            input_dim=3, qk_dim=128, v_dim=self.feat_dim,
            feature_dim=self.feat_dim, kernel_size=1, num_heads=4,
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

        # Pad to a multiple of vit_patch_size for the backbone, and also of
        # output_stride so the JAFAR output tiles cleanly.
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
            lr_feats, _ = self._backbone(tensor)  # (1, D, Hp/16, Wp/16)
            hr_feats = self._jafar(tensor, lr_feats, (out_h, out_w))
        grid = hr_feats[0].permute(1, 2, 0).cpu().numpy()  # (out_h, out_w, D)

        # Crop back to the unpadded image's token grid.
        n_h_orig = H // self.output_stride or 1
        n_w_orig = W // self.output_stride or 1
        if (n_h_orig, n_w_orig) != grid.shape[:2]:
            grid = grid[:n_h_orig, :n_w_orig]
        self._features = grid
        self._frame_shape = (H, W)

    def extract(self, frame_imgs, y_min, x_min, patch_size, cell_mask=None):
        # DINO features describe the patch + its image context globally —
        # cell_mask is intentionally NOT applied here.
        if self._features is None or self._frame_shape is None:
            return {c: np.nan for c in self.column_names}
        s = self.output_stride
        n_h, n_w = self._features.shape[:2]
        # Output tokens that overlap the patch [y_min, y_min+patch_size).
        gi_lo = max(0, int(y_min // s))
        gi_hi = min(n_h, int(-(-(y_min + patch_size) // s)))  # ceil
        gj_lo = max(0, int(x_min // s))
        gj_hi = min(n_w, int(-(-(x_min + patch_size) // s)))
        if gi_lo >= gi_hi or gj_lo >= gj_hi:
            return {c: np.nan for c in self.column_names}
        block = self._features[gi_lo:gi_hi, gj_lo:gj_hi]
        vec = block.reshape(-1, block.shape[-1]).mean(axis=0)
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
    patch_size: int = 14,
    progress_every: int = 2000,
    cell_mask_loader=None,
) -> pd.DataFrame:
    """Single pass over (fov, t). Each (t, fov) zarr frame is loaded once.

    Parameters
    ----------
    raw_zarr
        Zarr array of shape (T, P, C, H, W) (OmeZarrWriter direct mode).
    patches_long
        Long-form patches table from :func:`build_patches_long`.
    extractors
        Stack of :class:`FeatureExtractor`s. Order is preserved; columns
        from later extractors are appended after earlier ones.
    timepoints
        Which timepoints to process. Defaults to all T.
    n_imaging_channels
        How many leading channels to read per frame. Defaults to all
        channels in `raw_zarr`. Slicing here keeps the read off the
        stim-readout-only channels you don't need.
    patch_size
        Edge length of the patches in patches_long.
    progress_every
        Print a heartbeat after every N (fov, t, patch) triples processed.
    cell_mask_loader
        Optional ``callable(t, fov, particle) -> (H, W) bool`` returning
        the per-frame cell-pixel mask for the given tracked cell, or
        an all-False array if the cell isn't tracked at that frame.
        When provided, each patch's mask is passed to extractors via
        ``extract(..., cell_mask=...)``. Use :func:`make_cell_mask_loader`
        for the standard "seg label at this frame" lookup. Without a
        loader, extractors fall back to whole-patch (unmasked) means.

    Returns
    -------
    Long-form dataframe with columns: fov, particle, role, fov_timestep,
    and one column per name declared by each extractor.
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
        total=total_frames,
        unit="frame",
        desc="extracting",
        disable=progress_every == 0,
    )
    for fov in sorted(by_fov):
        patches = by_fov[fov]
        for t in timepoints:
            frame = np.asarray(raw_zarr[int(t), int(fov), :n_imaging_channels])
            # One prepare_frame per (t, fov) — DINO and any other model-
            # heavy extractors compute their per-frame state here, so
            # multiple patches on the same frame share the work.
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
                cell_mask = None
                if cell_mask_loader is not None:
                    cell_mask = cell_mask_loader(int(t), int(fov), int(p["particle"]))
                for ext in extractors:
                    rec.update(
                        ext.extract(
                            frame,
                            int(p["y_min"]),
                            int(p["x_min"]),
                            patch_size,
                            cell_mask=cell_mask,
                        )
                    )
                out_rows.append(rec)
            pbar.update(1)
    pbar.close()
    df = pd.DataFrame(out_rows)
    # If patches_long contained multiple patches per (cell, role) — e.g.
    # K control patches per cell — collapse the duplicates here by
    # averaging their feature values. With K=1 this is a no-op (each
    # group has one row).
    keys = ["fov", "particle", "role", "fov_timestep"]
    feat_cols = [c for c in df.columns if c not in keys]
    if feat_cols:
        df = df.groupby(keys, as_index=False)[feat_cols].mean()
    return df


# ----------------------------------------------------------------------------
# Reshape extracted features for plotting / merging
# ----------------------------------------------------------------------------


def pivot_roles(features_long: pd.DataFrame) -> pd.DataFrame:
    """Wide form: one row per (fov, particle, fov_timestep), columns
    suffixed with role.

    Output column names: ``<feature>_stim`` and ``<feature>_control``.
    Bookkeeping cols (fov, particle, fov_timestep) are kept as-is.
    """
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
    """Add ``<col>_fold`` columns: per (fov, particle, role) value divided by
    that cell+role's baseline aggregate across ``baseline_frames``.

    Idempotent: if ``features_long`` already carries ``<col>_baseline`` or
    ``<col>_fold`` columns (e.g. from a previous run loaded out of a
    parquet cache), they are dropped first so the merge below can
    re-create them cleanly instead of colliding into ``_x`` / ``_y``
    suffixes.

    method: "median" or "mean".
    """
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
    out = features_long.merge(
        base, on=["fov", "particle", "role"], how="left"
    )
    for c in columns:
        out[f"{c}_fold"] = out[c] / out[f"{c}_baseline"]
    return out


# ----------------------------------------------------------------------------
# Preview helpers
# ----------------------------------------------------------------------------


def plot_patch_overlays(
    raw_zarr,
    labels_perframe,
    patches_long: pd.DataFrame,
    t: int,
    *,
    n_cells: int = 6,
    channel_idx: int = 0,
    patch_size: int = 14,
    dot_diameter: float = 10,
    texture_window: int | None = None,
    dino_crop: int | None = None,
    crop: int = 200,
    rng_seed: int | None = None,
    cells: pd.DataFrame | None = None,
):
    """Stim + control patches (+ optional feature windows) on sample cells.

    For dialling in patch / window sizes before committing to the full
    extraction. Draws on the imaging channel ``channel_idx`` of ``raw`` at
    timepoint ``t``. ``texture_window`` and ``dino_crop`` are optional
    overlays for the GLCM and DINO crop sizes respectively.

    Returns (fig, axs).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pivot = patches_long.pivot_table(
        index=["fov", "particle"],
        columns="role",
        values=["y_min", "x_min"],
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
        sy = int(row[("y_min", "stim")])
        sx = int(row[("x_min", "stim")])
        cy = int(row[("y_min", "control")])
        cx = int(row[("x_min", "control")])

        center_y = sy + patch_size // 2
        center_x = sx + patch_size // 2
        y0 = max(0, center_y - half)
        y1 = min(H, y0 + crop)
        y0 = max(0, y1 - crop)
        x0 = max(0, center_x - half)
        x1 = min(W, x0 + crop)
        x0 = max(0, x1 - crop)

        img = np.asarray(raw_zarr[t, fov, channel_idx, y0:y1, x0:x1])
        lbl = np.asarray(labels_perframe[t, fov, y0:y1, x0:x1])
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.contour(lbl == particle, levels=[0.5], colors="lime", linewidths=0.6)

        # stim
        ax.add_patch(
            mpatches.Rectangle(
                (sx - x0, sy - y0),
                patch_size,
                patch_size,
                edgecolor="red",
                facecolor="none",
                linewidth=1.0,
            )
        )
        ax.add_patch(
            mpatches.Circle(
                (sx - x0 + (patch_size - 1) / 2, sy - y0 + (patch_size - 1) / 2),
                dot_diameter / 2,
                edgecolor="yellow",
                facecolor="none",
                linewidth=0.8,
            )
        )
        # control
        ax.add_patch(
            mpatches.Rectangle(
                (cx - x0, cy - y0),
                patch_size,
                patch_size,
                edgecolor="cyan",
                facecolor="none",
                linewidth=1.0,
            )
        )

        if texture_window:
            tx = sx - x0 + (patch_size - texture_window) / 2
            ty = sy - y0 + (patch_size - texture_window) / 2
            ax.add_patch(
                mpatches.Rectangle(
                    (tx, ty),
                    texture_window,
                    texture_window,
                    edgecolor="magenta",
                    facecolor="none",
                    linewidth=0.6,
                    linestyle="--",
                )
            )
        if dino_crop:
            dx = sx - x0 + (patch_size - dino_crop) / 2
            dy = sy - y0 + (patch_size - dino_crop) / 2
            ax.add_patch(
                mpatches.Rectangle(
                    (dx, dy),
                    dino_crop,
                    dino_crop,
                    edgecolor="orange",
                    facecolor="none",
                    linewidth=0.6,
                    linestyle=":",
                )
            )

        ax.set_title(f"fov={fov} particle={particle}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"t={t}  red=stim  cyan=control  yellow=dot"
        + ("  magenta=texture" if texture_window else "")
        + ("  orange=dino" if dino_crop else "")
    )
    fig.tight_layout()
    return fig, axs


# ----------------------------------------------------------------------------
# DINO PCA-RGB visualization
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
    """Per-cell PCA → (n_h, n_w, 3) RGB image.

    Unlike :func:`dino_features_pca_rgb`, which fits ONE PCA over every
    token in the frame, this fits a SEPARATE PCA per cell label. That
    way the components describe **within-cell** variation (nucleus vs
    cytoplasm, dense vs sparse regions, …) instead of being dominated by
    DINO's positional embedding artefacts (left vs right of the FOV,
    cells vs background, …).

    Each cell's three PCs are independently min-max scaled to [0, 1]
    (robust 2/98 percentile when the cell has ≥10 tokens; full range
    otherwise) and written into the output as RGB.

    Cells with fewer than ``min_tokens_per_cell`` (default 6) tokens are
    left as background — PCA on a handful of 384-D vectors is degenerate
    and would just emit noise colours.

    Parameters
    ----------
    features_grid : (n_h, n_w, D)
        DINO feature grid (the output of ``DINOv3JafarPatch.prepare_frame``).
    cell_labels : (n_h, n_w)
        Per-token cell-label image at the SAME resolution as ``features_grid``.
        0 means background (rendered as ``background_color``).
    """
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
    """Reduce a (n_h, n_w, D) DINO feature grid to a (n_h, n_w, 3) RGB image
    by PCA on the top-3 components, each scaled to [0, 1].

    Pass ``pca=`` an already-fit `sklearn.decomposition.PCA` to keep the
    component basis stable across frames (so frame-to-frame colour drift
    reflects real activation drift, not basis re-orientation). The fitted
    PCA is returned so you can reuse it on the next frame.

    With ``robust=True`` (default) per-component scaling clips to the
    2nd–98th percentile before normalising — handles ViT outlier tokens.
    """
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
# Responsiveness analysis
# ----------------------------------------------------------------------------


def per_cell_response_metric(
    features_long: pd.DataFrame,
    *,
    column: str,
    stim_frames: Iterable[int],
    agg: str = "max",
    role: str = "stim",
    fov_col: str = "fov",
    particle_col: str = "particle",
) -> pd.DataFrame:
    """Per-cell scalar response metric.

    For each (fov, particle), aggregate ``column`` over rows where role
    matches and ``fov_timestep`` is in ``stim_frames``. Default agg is
    ``max`` — use the peak fold change. ``mean`` / ``median`` are also
    accepted.
    """
    stim_frames = list(stim_frames)
    mask = (features_long["role"] == role) & (
        features_long["fov_timestep"].isin(stim_frames)
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
    """Stack DINO feature vectors per cell, averaged over ``frames``.

    Returns ``(X, cells_df)``:
      - ``X``: (N_cells, D) float32, one row per cell
      - ``cells_df``: (fov, particle) — row i of X corresponds to row i

    Average over baseline frames is the natural choice for a "static
    fingerprint" of the stim location pre-stimulation.
    """
    frames = list(frames)
    dino_cols = [c for c in features_long.columns if c.startswith(dino_prefix)]
    if not dino_cols:
        raise RuntimeError(
            f"no columns starting with {dino_prefix!r} in features_long; "
            "did you run the extraction with DINOv3Patch enabled?"
        )
    sub = features_long[
        (features_long["role"] == role)
        & (features_long["fov_timestep"].isin(frames))
    ]
    agg = (
        sub.groupby([fov_col, particle_col])[dino_cols].mean().reset_index()
    )
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
    """Binary responder classifier on DINO features.

    Fits a RandomForest + a LogisticRegression and returns both, plus
    cross-validated AUC and per-feature importance for each. The RF
    importance is impurity-based; the LR importance is |coef|.

    Returns a dict with keys: rf, logreg, scaler (or None), cv_auc_rf,
    cv_auc_lr, importance_rf, importance_lr.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X) if standardize else None
    Xs = scaler.transform(X) if scaler is not None else X

    rf = RandomForestClassifier(
        n_estimators=500, random_state=random_state, n_jobs=-1
    )
    lr = LogisticRegression(max_iter=2000, random_state=random_state)

    cv_auc_rf = cross_val_score(rf, Xs, y, cv=cv_folds, scoring="roc_auc").mean()
    cv_auc_lr = cross_val_score(lr, Xs, y, cv=cv_folds, scoring="roc_auc").mean()

    rf.fit(Xs, y)
    lr.fit(Xs, y)

    return {
        "rf": rf,
        "logreg": lr,
        "scaler": scaler,
        "cv_auc_rf": float(cv_auc_rf),
        "cv_auc_lr": float(cv_auc_lr),
        "importance_rf": np.asarray(rf.feature_importances_),
        "importance_lr": np.abs(lr.coef_[0]),
    }


def cluster_dino_features(
    X: np.ndarray,
    *,
    n_clusters: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """KMeans cluster labels for an (N, D) DINO matrix."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(X)


def fit_response_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int = 5,
    random_state: int = 0,
    standardize: bool = True,
):
    """Continuous-response regressor on DINO features.

    Predicts the actual response magnitude (e.g. peak fold change) per
    patch, not a binary responder label. Fits a RandomForest + a Ridge
    and reports cross-validated R² for each. The fitted RandomForest is
    the one you should hand to :func:`predict_response_map` — Ridge is
    here as a linear sanity baseline.

    Returns a dict with: rf, ridge, scaler (or None),
    cv_r2_rf, cv_r2_ridge, importance_rf.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X) if standardize else None
    Xs = scaler.transform(X) if scaler is not None else X

    rf = RandomForestRegressor(
        n_estimators=500, random_state=random_state, n_jobs=-1
    )
    ridge = Ridge(alpha=1.0, random_state=random_state)

    cv_r2_rf = cross_val_score(rf, Xs, y, cv=cv_folds, scoring="r2").mean()
    cv_r2_ridge = cross_val_score(ridge, Xs, y, cv=cv_folds, scoring="r2").mean()

    rf.fit(Xs, y)
    ridge.fit(Xs, y)

    return {
        "rf": rf,
        "ridge": ridge,
        "scaler": scaler,
        "cv_r2_rf": float(cv_r2_rf),
        "cv_r2_ridge": float(cv_r2_ridge),
        "importance_rf": np.asarray(rf.feature_importances_),
    }


def predict_response_map(
    model,
    features_grid: np.ndarray,
    *,
    scaler=None,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply a fitted regressor to every ViT token, returning a 2-D
    "predicted response" heatmap.

    ``features_grid``: (n_h, n_w, D) DINO output from
    ``DINOv3Patch.prepare_frame``.
    ``scaler``: the StandardScaler from :func:`fit_response_regressor`
    (apply to keep the input distribution identical to training).
    ``image_shape``: optional (H, W) — if given, the (n_h, n_w) map is
    nearest-neighbour-upsampled to pixel resolution so it overlays the
    raw image directly.

    Each pixel's value is the model's prediction for *what the response
    would be if the stim patch were placed at that location*.
    """
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


def dino_feature_map(
    features_grid: np.ndarray,
    feature_index: int,
    *,
    image_shape: tuple[int, int] | None = None,
    robust: bool = True,
) -> np.ndarray:
    """Spatial activation map for a single DINO feature.

    Returns a 2-D array of the feature's value at each token. If
    ``image_shape=(H, W)`` is given, the (n_h, n_w) map is repeated up to
    pixel resolution by nearest-neighbour so it can be overlaid on the
    raw image.

    ``robust`` (default True) clips to 2/98 percentile so a single noisy
    token doesn't squash the colour scale.
    """
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
