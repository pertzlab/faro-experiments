"""Components specific to this experiment.

``RandomStimPerCellCircle`` is the stimulator the experiment notebook uses.
It lives here rather than in faro because no second experiment needs it
yet. The analysis notebook imports the same class through ``stim_analysis``.
"""

"""Per-cell random circular stimulation.

For every segmented cell, pick one random pixel inside the cell where a
``DIAMETER``-pixel disk fits completely within the cell boundary (computed
via the cell's distance-to-boundary transform). The disk is stamped onto
the stim mask.

Differences vs ``RandomStimPerCell14pxPatches``:

* No grid alignment — the centroid can sit at any pixel inside the cell.
  (This was forced by DINOv2's 14 px ViT patch; with JAFAR upsampling the
  features are no longer pinned to the grid.)
* Stim region is a circle defined by its (y, x) centre and radius, not a
  bounding box. Stored as ``stim_center_y`` / ``stim_center_x`` /
  ``stim_radius`` columns.
* The centre + radius are appended directly to the tracking dataframe
  (in place) on the stim-frame rows of the corresponding tracked particle,
  so the analysis notebook does not need a post-acquisition merge step.

As before, the selection is taken **once per FOV** (at the first stim
frame) and reused for every subsequent stim frame in that FOV so the
illuminated spots stay fixed in image space across a stim block. Cells
that aren't tracked at the first stim frame (no particle id) are dropped
— there's no stable handle for them across frames.
"""

import threading

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from faro.stimulation.base import StimWithPipeline


class RandomStimPerCellCircle(StimWithPipeline):
    DIAMETER = 7  # px; circular stim region's diameter

    def __init__(self, *, seed: int = 0):
        self._base_seed = int(seed)
        self._radius = self.DIAMETER / 2.0
        # Records list kept for debugging / parity with the patch stimulator;
        # the analysis path doesn't need it because the (centre, radius) is
        # also written directly to the tracking dataframe.
        self.records: list[dict] = []
        self._lock = threading.Lock()
        # Per-FOV cache of {particle_id: (cy, cx)}. Populated on the first
        # stim frame for that FOV and reused on every subsequent stim
        # frame so the illuminated spots stay fixed in image space.
        self._fov_selections: dict[int, dict[int, tuple[int, int]]] = {}

        # Pre-compute the disk template — a (2r+1, 2r+1) uint8 array.
        rr = int(np.ceil(self._radius))
        yy, xx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
        self._disk = ((yy * yy + xx * xx) <= self._radius * self._radius).astype(
            np.uint8
        ) * 255
        self._disk_r = rr

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _rng_for(self, metadata):
        fov = int(metadata.get("fov", 0)) if metadata else 0
        seed = (self._base_seed * 2_654_435_761 + fov * 1_000_003) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _select_centroid_per_cell(self, labels: np.ndarray, rng) -> dict:
        """For each cell label, return (cy, cx) where the disk fits inside.

        Uses the per-cell distance transform: any pixel whose distance to
        the cell boundary is at least the disk's radius can host the
        centre. One such pixel is picked uniformly at random per cell.
        Cells too thin to contain the disk anywhere are dropped.
        """
        per_cell: dict[int, tuple[int, int]] = {}
        labels_arr = np.asarray(labels)
        for label in np.unique(labels_arr):
            if label == 0:
                continue
            mask = labels_arr == label
            edt = distance_transform_edt(mask)
            ys, xs = np.where(edt >= self._radius)
            if len(ys) == 0:
                continue
            idx = int(rng.integers(len(ys)))
            per_cell[int(label)] = (int(ys[idx]), int(xs[idx]))
        return per_cell

    def _stamp_disk(self, mask: np.ndarray, cy: int, cx: int) -> None:
        H, W = mask.shape
        r = self._disk_r
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        # Clip to image bounds. The disk fits inside the cell, which is
        # inside the image, so clipping is only defensive.
        ay0, ay1 = max(0, y0), min(H, y1)
        ax0, ax1 = max(0, x0), min(W, x1)
        dy0, dx0 = ay0 - y0, ax0 - x0
        tile = self._disk[dy0 : dy0 + (ay1 - ay0), dx0 : dx0 + (ax1 - ax0)]
        np.maximum(mask[ay0:ay1, ax0:ax1], tile, out=mask[ay0:ay1, ax0:ax1])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        meta = metadata or {}
        fov = meta.get("fov")
        fov_timestep = meta.get("fov_timestep", meta.get("timestep"))

        cache_key = int(fov) if fov is not None else 0
        if cache_key not in self._fov_selections:
            rng = self._rng_for(meta)
            per_seg_label = self._select_centroid_per_cell(labels, rng)
            # Re-key the selection by *particle id* so the same cell can
            # be looked up at later stim frames after Cellpose has
            # renumbered its seg labels. Cells without a particle id at
            # this frame are dropped.
            per_particle: dict[int, tuple[int, int]] = {}
            if (
                tracks is not None
                and not tracks.empty
                and "particle" in tracks.columns
                and "label" in tracks.columns
            ):
                frame_rows = self._filter_to_frame(tracks, fov, fov_timestep)
                if "particle" in frame_rows.columns:
                    valid = frame_rows.dropna(subset=["particle"])
                    seg_to_part = dict(
                        zip(
                            valid["label"].astype(int),
                            valid["particle"].astype(int),
                        )
                    )
                    for seg_label, (cy, cx) in per_seg_label.items():
                        particle = seg_to_part.get(int(seg_label))
                        if particle is not None:
                            per_particle[int(particle)] = (cy, cx)
            else:
                # No tracking column available — fall back to seg labels.
                per_particle = {int(k): v for k, v in per_seg_label.items()}
            self._fov_selections[cache_key] = per_particle

        per_particle = self._fov_selections[cache_key]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        for cy, cx in per_particle.values():
            self._stamp_disk(stim_mask, cy, cx)

        # Append stim centre + radius directly to the tracks df for cells
        # that have a stim spot at this (fov, fov_timestep). The pipeline
        # uses the mutated df_tracked downstream (feature extraction, then
        # parquet save), so the columns survive into exp_data.parquet
        # without a post-process merge.
        if tracks is not None and not tracks.empty:
            for col in ("stim_center_y", "stim_center_x", "stim_radius"):
                if col not in tracks.columns:
                    tracks[col] = np.nan
            if "particle" in tracks.columns:
                base_mask = self._frame_mask(tracks, fov, fov_timestep)
                rows: list[dict] = []
                for particle, (cy, cx) in per_particle.items():
                    cell_mask = base_mask & (tracks["particle"] == particle)
                    tracks.loc[cell_mask, "stim_center_y"] = float(cy)
                    tracks.loc[cell_mask, "stim_center_x"] = float(cx)
                    tracks.loc[cell_mask, "stim_radius"] = float(self._radius)
                    rows.append(
                        {
                            "fov": fov,
                            "fov_timestep": fov_timestep,
                            "particle": int(particle),
                            "stim_center_y": float(cy),
                            "stim_center_x": float(cx),
                            "stim_radius": float(self._radius),
                        }
                    )
                if rows:
                    with self._lock:
                        self.records.extend(rows)

        return stim_mask, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_mask(tracks: pd.DataFrame, fov, fov_timestep) -> pd.Series:
        m = pd.Series(True, index=tracks.index)
        if "fov_timestep" in tracks.columns and fov_timestep is not None:
            m &= tracks["fov_timestep"] == fov_timestep
        if "fov" in tracks.columns and fov is not None:
            m &= tracks["fov"] == fov
        return m

    def _filter_to_frame(
        self, tracks: pd.DataFrame, fov, fov_timestep
    ) -> pd.DataFrame:
        return tracks[self._frame_mask(tracks, fov, fov_timestep)]

    def to_dataframe(self) -> pd.DataFrame:
        """Per-cell stim centroid records accumulated during the run.

        Kept for parity with ``RandomStimPerCell14pxPatches`` and for
        debugging; the analysis path reads the same info straight off
        ``exp_data.parquet`` because the columns are mutated onto the
        tracks dataframe in :meth:`get_stim_mask`.
        """
        with self._lock:
            return pd.DataFrame(list(self.records))
