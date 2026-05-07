import json
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection
from shapely.geometry import Polygon
from skimage.draw import polygon as _sk_polygon
from skimage.morphology import skeletonize as _skeletonize

# ---------------------------------------------------------------------------
# WGS-84 ellipsoid constants
# ---------------------------------------------------------------------------
_WGS84_A  = 6378137.0           # semi-major axis (metres)
_WGS84_F  = 1 / 298.257223563   # flattening
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2  # first eccentricity squared


# ---------------------------------------------------------------------------
# Phase 1 – data loading & coordinate conversion
# ---------------------------------------------------------------------------

def extract_cats_from_geojson(path, categories):
    """Return features whose NAME property matches any entry in *categories*.

    Matching is case-insensitive.  Each returned item is a dict::

        {
            "shape_id":    str,
            "name":        str,   # original case from file
            "coordinates": list,  # outer ring – list of [lon, lat, alt]
        }
    """
    lower_cats = {c.lower() for c in categories}
    features = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for feat in data["features"]:
        props = feat["properties"]
        if props.get("NAME", "").lower() in lower_cats:
            outer_ring = feat["geometry"]["coordinates"][0]
            features.append({
                "shape_id":    props["shape_id"],
                "name":        props["NAME"],
                "coordinates": outer_ring,
            })
    return features


def _enu_scale_at(lat_rad):
    """Return (m_per_rad_north, m_per_rad_east) for WGS-84 at *lat_rad*."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    denom = math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    M = _WGS84_A * (1.0 - _WGS84_E2) / denom ** 3   # meridional radius of curvature
    N = _WGS84_A / denom                              # prime vertical radius
    return M, N * cos_lat  # (north scale, east scale) in m/rad


def convert_geojson_to_cartesian(features):
    """Convert per-vertex (lon, lat, alt) to local ENU metres.

    Each polygon uses its own vertex centroid as the local origin so that
    coordinates are in the range of centimetres-to-metres appropriate for
    rock-art panel features.  Altitude is already in metres; longitude and
    latitude are converted using the WGS-84 ellipsoid.

    Returns a new list of feature dicts, each extended with::

        "cartesian": np.ndarray, shape (N, 3)  – columns: east, north, up (m)
        "origin":    (lon0, lat0, alt0)         – the local origin used
    """
    DEG2RAD = math.pi / 180.0
    result = []
    for feat in features:
        coords = feat["coordinates"]
        # Drop closing vertex if the ring is explicitly closed
        verts = coords[:-1] if coords[0] == coords[-1] else coords

        lons = np.array([v[0] for v in verts])
        lats = np.array([v[1] for v in verts])
        alts = np.array([v[2] for v in verts])

        lon0 = lons.mean()
        lat0 = lats.mean()
        alt0 = alts.mean()

        M, N_cos = _enu_scale_at(lat0 * DEG2RAD)

        east  = (lons - lon0) * DEG2RAD * N_cos
        north = (lats - lat0) * DEG2RAD * M
        up    = alts - alt0

        cartesian = np.column_stack([east, north, up])
        result.append({**feat, "cartesian": cartesian, "origin": (lon0, lat0, alt0)})
    return result


# ---------------------------------------------------------------------------
# Phase 2 – PCA projection to 2D
# ---------------------------------------------------------------------------

def pca_reduce_to_2d(cartesian_features):
    """Project each 3D polygon onto its best-fit plane via PCA.

    For each feature the three principal components are found from the
    centred vertex cloud.  The polygon is projected onto the plane spanned
    by PC1 and PC2 (the two axes of greatest variance), which corresponds
    to the best-fit plane through the vertices.

    Returns a new list of feature dicts, each extended with::

        "cartesian_2d":    np.ndarray, shape (N, 2)  – projected coordinates
        "pca_components":  np.ndarray, shape (3, 3)  – rows are PC1, PC2, PC3
        "explained_var":   np.ndarray, shape (3,)    – explained variance ratios
        "pc3_flag":        bool  – True if PC3 explains >5 % of variance
    """
    result = []
    for feat in cartesian_features:
        pts = feat["cartesian"]          # (N, 3)
        centred = pts - pts.mean(axis=0)

        # SVD of the centred cloud: V rows are principal components
        _, s, Vt = np.linalg.svd(centred, full_matrices=False)
        var = s ** 2
        explained = var / var.sum()

        # Project onto the first two PCs
        coords_2d = centred @ Vt[:2].T   # (N, 2)

        result.append({
            **feat,
            "cartesian_2d":   coords_2d,
            "pca_components": Vt,          # (3, 3) – rows PC1, PC2, PC3
            "explained_var":  explained,
            "pc3_flag":       bool(explained[2] > 0.05),
        })
    return result


# ---------------------------------------------------------------------------
# Phase 3 – FPD shape normalisation
# ---------------------------------------------------------------------------

def resample_polygon(pts, n):
    """Resample a closed 2D polygon to exactly *n* equally-spaced points.

    Uses arc-length parameterisation: cumulative chord lengths are computed,
    then *n* equally-spaced arc positions are interpolated.  The polygon is
    treated as closed (last point connects back to first).

    Parameters
    ----------
    pts : np.ndarray, shape (M, 2)
    n   : int – target number of points

    Returns
    -------
    np.ndarray, shape (n, 2)
    """
    # Close the ring temporarily for arc computation
    closed = np.vstack([pts, pts[0]])
    diffs  = np.diff(closed, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)          # (M,)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)]) # (M+1,)
    total = arc[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1))
    targets = np.linspace(0.0, total, n, endpoint=False)
    resampled = np.column_stack([
        np.interp(targets, arc, closed[:, 0]),
        np.interp(targets, arc, closed[:, 1]),
    ])
    return resampled


def _fpd_signature(pts):
    """Compute the Farthest Point Distance signature for a resampled polygon.

    For each boundary point u::

        FPD(u) = dist(u, centroid) + dist(farthest_point_from_u, centroid)

    This is a real-valued, translation-invariant 1-D signature.
    Mirror-image shapes produce identical signatures (desirable here).

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)  – already resampled, centred not required

    Returns
    -------
    np.ndarray, shape (N,)
    """
    centroid = pts.mean(axis=0)
    # dist from each point to centroid
    d_to_cent = np.linalg.norm(pts - centroid, axis=1)   # (N,)
    # pairwise distances: (N, N)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    pairwise = np.linalg.norm(diff, axis=2)               # (N, N)
    # for each point, the farthest point
    fp_idx = np.argmax(pairwise, axis=1)                  # (N,)
    d_fp_to_cent = d_to_cent[fp_idx]                      # (N,)
    return d_to_cent + d_fp_to_cent


def compute_fpd_descriptors(features_2d, n_resample=128, n_descriptors=63):
    """Compute normalised FPD Fourier descriptors for each 2D feature.

    Pipeline (following El-ghazal et al. 2009):

    1. Resample polygon to *n_resample* equally-spaced points.
    2. Compute FPD signature (real-valued, translation-invariant).
    3. Apply DFT.
    4. Scale-normalise: divide magnitudes of FD_1 … FD_{n_descriptors} by FD_0.
    5. Discard phase (rotation and starting-point invariant).

    Returns a new list of feature dicts, each extended with::

        "resampled":    np.ndarray, shape (n_resample, 2)
        "fpd_sig":      np.ndarray, shape (n_resample,)  – raw FPD signature
        "fpd_desc":     np.ndarray, shape (n_descriptors,)  – normalised FDs
    """
    result = []
    for feat in features_2d:
        pts = resample_polygon(feat["cartesian_2d"], n_resample)
        sig = _fpd_signature(pts)

        fd  = np.fft.fft(sig)                # complex DFT, length n_resample
        fd0 = np.abs(fd[0])                  # DC component (average energy)
        if fd0 == 0:
            desc = np.zeros(n_descriptors)
        else:
            # Take first n_descriptors non-DC magnitudes, normalise by FD0
            desc = np.abs(fd[1:n_descriptors + 1]) / fd0

        result.append({
            **feat,
            "resampled": pts,
            "fpd_sig":   sig,
            "fpd_desc":  desc,
        })
    return result


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def visualise_samples(features_2d, n, title="2D polygon samples"):
    """Plot *n* random samples from *features_2d* in an adaptive grid.

    Each feature must have a ``"cartesian_2d"`` key containing an (N, 2)
    array (added by ``pca_reduce_to_2d``).  The grid has ``ceil(sqrt(n))``
    columns and enough rows to fit all samples.
    """
    sample = random.sample(features_2d, min(n, len(features_2d)))
    cols = math.ceil(math.sqrt(len(sample)))
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for ax, feat in zip(axes, sample):
        pts = feat["cartesian_2d"]
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], lw=1)
        ax.set_aspect("equal")
        ax.set_title(feat["name"], fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualise_samples_3d(features_cart, n, title="3D polygon samples (ENU metres)"):
    """Plot *n* random samples from *features_cart* as 3D line plots.

    Each feature must have a ``"cartesian"`` key containing an (N, 3) array.
    """
    sample = random.sample(features_cart, min(n, len(features_cart)))
    cols = math.ceil(math.sqrt(len(sample)))
    rows = math.ceil(len(sample) / cols)
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for idx, feat in enumerate(sample, 1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        pts = feat["cartesian"]
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], lw=1)
        ax.set_title(feat["name"], fontsize=8)
        ax.set_xlabel("E (m)", fontsize=6)
        ax.set_ylabel("N (m)", fontsize=6)
        ax.set_zlabel("U (m)", fontsize=6)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Phase 4 – completeness filtering
# ---------------------------------------------------------------------------

# Normalised-height bands (fraction of oriented bounding-box height, 0=bottom 1=top)
REGION_BANDS = {
    "head":  (0.65, 1.00),
    "torso": (0.35, 0.70),
    "groin": (0.20, 0.50),
    "legs":  (0.00, 0.35),
}


def simplify_polygon(pts, epsilon=0.02):
    """Simplify a closed 2-D polygon with Douglas-Peucker via shapely.

    Returns simplified vertices as an (M, 2) array, or the original if
    simplification produces a degenerate result.
    """
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    simplified = poly.simplify(epsilon, preserve_topology=True)
    if simplified.is_empty or simplified.geom_type != "Polygon":
        return pts
    coords = np.array(simplified.exterior.coords)[:-1]   # drop closing duplicate
    return coords if len(coords) >= 4 else pts


def orient_polygon_upright(pts):
    """Rotate a polygon so its principal axis aligns with Y, narrower end at top.

    Uses SVD to find the principal axis, rotates to align it with Y, then
    checks cross-section widths at the 20 % extremes to flip if needed so
    the narrower end (head) ends up at the top.
    """
    centred = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    principal = Vt[0]
    angle = np.arctan2(principal[0], principal[1])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = centred @ R.T

    ymin, ymax = rotated[:, 1].min(), rotated[:, 1].max()
    h = (ymax - ymin) or 1.0
    top_pts = rotated[rotated[:, 1] > ymax - 0.2 * h]
    bot_pts = rotated[rotated[:, 1] < ymin + 0.2 * h]
    top_w = float(np.ptp(top_pts[:, 0])) if len(top_pts) > 1 else 0.0
    bot_w = float(np.ptp(bot_pts[:, 0])) if len(bot_pts) > 1 else 0.0
    if bot_w < top_w:
        rotated[:, 1] *= -1
    return rotated


def extract_protrusion_keypoints(pts, prox_ratio=0.85):
    """Return vertices that are local maxima of centroid-distance.

    Uses a ±1-vertex window.  Only vertices whose centroid-distance exceeds
    ``prox_ratio * median`` are considered.  A value below 1.0 means points
    slightly below the median are still eligible, which is necessary for
    figures whose limbs are close to the body axis after DP simplification.
    """
    centroid  = pts.mean(axis=0)
    dists     = np.linalg.norm(pts - centroid, axis=1)
    threshold = np.median(dists) * prox_ratio
    n = len(pts)
    kp_idx = [
        i for i in range(n)
        if dists[i] >= threshold
        and dists[i] >= dists[(i - 1) % n]
        and dists[i] >= dists[(i + 1) % n]
    ]
    return pts[kp_idx] if kp_idx else np.empty((0, 2))


def classify_keypoints(kps, bbox):
    """Assign keypoints to anatomical bands by normalised Y position.

    ``bbox`` = (xmin, ymin, xmax, ymax) of the oriented polygon.
    Returns ``{region: [np.ndarray, ...]}``.  Each keypoint is assigned
    to the first matching band in ``REGION_BANDS`` order.
    """
    xmin, ymin, xmax, ymax = bbox
    h = (ymax - ymin) if ymax > ymin else 1.0
    classified = {r: [] for r in REGION_BANDS}
    for kp in kps:
        y_norm = (kp[1] - ymin) / h
        for region, (lo, hi) in REGION_BANDS.items():
            if lo <= y_norm <= hi:
                classified[region].append(kp)
                break
    return classified


def _is_female(name):
    return "female" in name.lower()


def _is_male(name):
    return "male" in name.lower() and not _is_female(name)


def completeness_report(feature, epsilon=0.02, prox_ratio=1.02):
    """Run simplify → orient → keypoint → classify → score for one feature.

    Parameters
    ----------
    feature   : dict with ``cartesian_2d`` (N, 2) and ``name`` / ``shape_id``
    epsilon   : Douglas-Peucker tolerance in ENU metres
    prox_ratio: protrusion threshold as a multiple of median centroid-distance

    Returns a dict with keys:
        simplified_pts, oriented_pts, keypoints, classified,
        parts, score, complete, shape_id, name
    """
    pts  = feature["cartesian_2d"]
    name = feature.get("name", "")

    simplified = simplify_polygon(pts, epsilon)
    oriented   = orient_polygon_upright(simplified)
    kps        = extract_protrusion_keypoints(oriented, prox_ratio)

    if len(kps) == 0:
        return dict(simplified_pts=simplified, oriented_pts=oriented,
                    keypoints=kps, classified={},
                    parts={}, score=0, complete=False,
                    shape_id=feature["shape_id"], name=name)

    bbox       = (oriented[:, 0].min(), oriented[:, 1].min(),
                  oriented[:, 0].max(), oriented[:, 1].max())
    classified = classify_keypoints(kps, bbox)

    has_head  = len(classified["head"])  >= 1
    has_arms  = (len(classified["torso"]) + len(classified["groin"])) >= 1
    has_legs  = len(classified["legs"])  >= 1
    has_torso = len(classified["torso"]) >= 1

    parts = {"head": has_head, "arms": has_arms, "legs": has_legs, "torso": has_torso}

    if _is_female(name):
        parts["breasts"] = len(classified["torso"]) >= 3
    if _is_male(name):
        parts["penis"] = len(classified["groin"]) >= 1

    score    = sum(parts.values())
    # Complete = at least 3 of the 4 core parts detected.
    # Allows for one part merging into the outline after DP simplification.
    core_detected = sum([has_head, has_arms, has_legs, has_torso])
    complete = core_detected >= 3

    return dict(
        simplified_pts=simplified,
        oriented_pts=oriented,
        keypoints=kps,
        classified=classified,
        parts=parts,
        score=score,
        complete=complete,
        shape_id=feature["shape_id"],
        name=name,
    )


# ---------------------------------------------------------------------------
# Phase 4 – skeleton-based completeness filtering
# ---------------------------------------------------------------------------

def _rasterise_polygon(pts, size=128):
    """Rasterise a closed 2-D polygon to a bool (size × size) mask."""
    mn  = pts.min(axis=0)
    mx  = pts.max(axis=0)
    rng = (mx - mn).copy()
    rng[rng == 0] = 1.0
    col = (pts[:, 0] - mn[0]) / rng[0] * (size - 5) + 2
    row = (size - 3) - (pts[:, 1] - mn[1]) / rng[1] * (size - 5)
    mask = np.zeros((size, size), dtype=bool)
    rr, cc = _sk_polygon(row, col, shape=(size, size))
    mask[rr, cc] = True
    return mask


def _skeleton_stats(skel):
    """Return (endpoints, branch_pts) lists of (row, col) for a skeleton."""
    rows, cols = np.where(skel)
    h, w = skel.shape
    endpoints, branch_pts = [], []
    for r, c in zip(rows, cols):
        degree = int(skel[max(r-1,0):r+2, max(c-1,0):c+2].sum()) - 1
        if degree == 1:
            endpoints.append((r, c))
        elif degree >= 3:
            branch_pts.append((r, c))
    return endpoints, branch_pts


def skeleton_completeness_report(feature, raster_size=128,
                                  head_band=(0.65, 1.00),
                                  arm_band=(0.30, 0.75),
                                  leg_band=(0.00, 0.38),
                                  min_branches=1):
    """Rasterise → skeletonise → orient upright → classify endpoints anatomically.

    After skeletonisation the endpoint pixel coordinates are SVD-oriented so
    the principal body axis aligns with Y (narrow end = head at top), then
    each endpoint is classified by its normalised height:

        head  : y_norm in head_band   (top of figure)
        arms  : y_norm in arm_band    (mid-height, lateral extremes)
        legs  : y_norm in leg_band    (bottom of figure)

    A figure is **complete** when:
        - ≥1 head endpoint
        - ≥1 arm endpoint
        - ≥1 leg endpoint
        - ≥min_branches branch points (body junctions)

    Parameters
    ----------
    feature     : dict with ``cartesian_2d``, ``shape_id``, ``name``
    raster_size : pixel resolution (default 128)
    head_band   : (lo, hi) normalised-Y range for head endpoints
    arm_band    : (lo, hi) normalised-Y range for arm endpoints
    leg_band    : (lo, hi) normalised-Y range for leg endpoints
    min_branches: branch points required (body junctions exist)

    Returns a dict with keys:
        mask, skeleton, endpoints, branches,
        ep_head, ep_arms, ep_legs,
        n_endpoints, n_branches, complete, shape_id, name
    """
    pts  = feature["cartesian_2d"]
    name = feature.get("name", "")
    mask = _rasterise_polygon(pts, raster_size)
    skel = _skeletonize(mask)
    eps, brs = _skeleton_stats(skel)

    if len(eps) == 0:
        return dict(mask=mask, skeleton=skel,
                    endpoints=eps, branches=brs,
                    ep_head=[], ep_arms=[], ep_legs=[],
                    n_endpoints=0, n_branches=len(brs),
                    complete=False,
                    shape_id=feature["shape_id"], name=name)

    # Convert (row, col) endpoints to (x, y) with Y pointing up
    ep_arr = np.array(eps, dtype=float)          # (M, 2)  col0=row, col1=col
    xy = np.column_stack([ep_arr[:, 1],          # x = pixel col
                          raster_size - ep_arr[:, 0]])  # y = flipped row

    # SVD-orient so principal axis aligns with Y, narrower end at top
    centred = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    angle = np.arctan2(Vt[0, 0], Vt[0, 1])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot = centred @ R.T

    ymin, ymax = rot[:, 1].min(), rot[:, 1].max()
    h = (ymax - ymin) or 1.0
    top_pts = rot[rot[:, 1] > ymax - 0.2 * h]
    bot_pts = rot[rot[:, 1] < ymin + 0.2 * h]
    top_w = float(np.ptp(top_pts[:, 0])) if len(top_pts) > 1 else 0.0
    bot_w = float(np.ptp(bot_pts[:, 0])) if len(bot_pts) > 1 else 0.0
    if bot_w < top_w:          # narrower end is at bottom — flip so head is up
        rot[:, 1] *= -1
        ymin, ymax = rot[:, 1].min(), rot[:, 1].max()
        h = (ymax - ymin) or 1.0

    # Classify each endpoint by normalised Y position
    y_norm = (rot[:, 1] - ymin) / h
    ep_head = [eps[i] for i, y in enumerate(y_norm) if head_band[0] <= y <= head_band[1]]
    ep_arms = [eps[i] for i, y in enumerate(y_norm) if arm_band[0]  <= y <= arm_band[1]]
    ep_legs = [eps[i] for i, y in enumerate(y_norm) if leg_band[0]  <= y <= leg_band[1]]

    complete = (len(ep_head) >= 1 and
                len(ep_arms) >= 1 and
                len(ep_legs) >= 1 and
                len(brs)     >= min_branches)

    return dict(
        mask=mask, skeleton=skel,
        endpoints=eps, branches=brs,
        ep_head=ep_head, ep_arms=ep_arms, ep_legs=ep_legs,
        n_endpoints=len(eps), n_branches=len(brs),
        complete=complete,
        shape_id=feature["shape_id"], name=name,
    )