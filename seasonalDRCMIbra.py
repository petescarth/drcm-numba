#!/usr/bin/env python
"""
seasonalDRCMIbra.py

IBRA-Constrained Seasonal DRCM (Dynamic Reference Cover Method) Processing.

Compares seasonal fractional groundcover against long-term reference statistics
strictly constrained to matching IBRA biogeographic regions (1..255, covering
IBRA 7.0 regions 1..89), or using a single global histogram (--no-ibra).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
import re
import sys
import time

try:
    import resource
except ImportError:
    resource = None

import numpy as np
from numba import njit, prange
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

from refpix_numba_ibra import refpixelfilter_ibra
from refpix_numba import refpixelfilternodata


def format_time(seconds):
    """Format elapsed seconds into a compact human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute IBRA-constrained Seasonal DRCM (Difference Ground Cover & Reference Cover)."
    )
    parser.add_argument(
        "input_groundcover",
        help="Input seasonal bare groundcover GeoTIFF (e.g. lztmre_aus_m202606202608_dp8a2_vinterim.tif)",
    )
    parser.add_argument(
        "--percentile-file",
        default="lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif",
        help="2-band reference raster (Band 1: Stats, Band 2: IBRA Code). Default: lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif",
    )
    parser.add_argument(
        "--ibra-raster",
        default=None,
        help="Optional separate rasterized IBRA raster if percentile-file is single-band.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output GeoTIFF path (or directory ending with '/'). If not specified, derived automatically from the input filename.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1501,
        help="Comparison window size in pixels (must be odd, default: 1501)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4096,
        help="Number of lines to process per chunk (default: 4096)",
    )
    parser.add_argument(
        "--lowerquantile",
        type=float,
        default=0.90,
        help="Lower quantile bound for finding high persistent cover reference sites (default: 0.90)",
    )
    parser.add_argument(
        "--upperquantile",
        type=float,
        default=0.95,
        help="Upper quantile bound for finding high persistent cover reference sites (default: 0.95)",
    )
    parser.add_argument(
        "--minvalpercent",
        type=float,
        default=0.15,
        help="Minimum fraction of valid pixels within comparison window in matching IBRA (default: 0.15)",
    )
    parser.add_argument(
        "--min-obs-percent",
        type=float,
        default=0.50,
        help="Minimum seasonal cloud-free fraction of static reference pixels (v_obs / perc_tot) for sparse-reference recovery (0 disables, default: 0.50)",
    )
    parser.add_argument(
        "--sparse-min-valpercent",
        type=float,
        default=0.02,
        help="Minimum static reference pixel density (perc_tot / N_window) eligible for sparse-reference recovery (default: 0.02)",
    )
    parser.add_argument(
        "--min-ref-samples",
        type=int,
        default=500,
        help="Minimum number of valid seasonal observations required inside [lowerquantile, upperquantile] (default: 500)",
    )
    parser.add_argument(
        "--test-chip",
        nargs=4,
        type=int,
        metavar=("XOFF", "YOFF", "XSIZE", "YSIZE"),
        help="Optional test chip window [xoff, yoff, xsize, ysize] for unpadded micro-subsetting",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Projected bounding box [minx miny maxx maxy] in input raster CRS (halo-padded reads so edge pixels match full-extent runs)",
    )
    parser.add_argument(
        "--geojson",
        default=None,
        help="Path to vector AOI file (e.g. GeoJSON) whose envelope subsets output cleanly with halo padding",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of Numba parallel worker threads to use (default: available CPUs / container cgroup quota)",
    )
    parser.add_argument(
        "--overviews",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
        help="Overview levels to calculate using average resampling (default: 4 8 16 32)",
    )
    parser.add_argument(
        "--no-overviews",
        action="store_true",
        help="Skip calculating internal overviews on completion",
    )
    parser.add_argument(
        "--no-ibra",
        action="store_true",
        help="Do not constrain by IBRA regions; calculate reference statistics using a single global histogram.",
    )
    parser.add_argument(
        "--bare-encoding",
        choices=("auto", "dp8a2", "dixa2"),
        default="auto",
        help="Input bare ground scaling stage (default: auto, inferring dp8 vs dix from filename)",
    )
    return parser.parse_args(argv)


_STAGE_DIX_RE = re.compile(r"(?:^|[_\-.])dix", re.IGNORECASE)
_STAGE_DP8_RE = re.compile(r"(?:^|[_\-.])dp8", re.IGNORECASE)


def infer_bare_encoding_from_filename(input_path):
    """Infer bare ground stage ('dixa2' vs 'dp8a2') from pipeline stage tokens in filename."""
    base = os.path.basename(str(input_path or ""))
    if _STAGE_DIX_RE.search(base):
        return "dixa2"
    if _STAGE_DP8_RE.search(base):
        return "dp8a2"
    return None


def derive_output_name(input_path, no_ibra=False):
    """Derive a standard DRCM output filename from an input filename."""
    base = os.path.basename(input_path)
    root, ext = os.path.splitext(base)
    suffix = "doda2" if no_ibra else "doda2_ibra"
    generic = "_drcm" if no_ibra else "_drcm_ibra"
    if "dixa2" in root:
        out_root = root.replace("dixa2", suffix)
    elif "dp8a2" in root:
        out_root = root.replace("dp8a2", suffix)
    else:
        out_root = f"{root}{generic}"
    return f"{out_root}{ext}"


@njit(parallel=True)
def _scan_valid_minmax_u8_2d(arr, nodata_i16):
    """Scan uint8 array for min/max candidate bare values in [0, 200] excluding nodata."""
    h, w = arr.shape
    row_has = np.zeros(h, dtype=np.uint8)
    row_min = np.full(h, 255, dtype=np.uint8)
    row_max = np.zeros(h, dtype=np.uint8)
    for r in prange(h):
        has_v = np.uint8(0)
        lo = np.uint8(255)
        hi = np.uint8(0)
        for c in range(w):
            v = arr[r, c]
            if v <= 200 and np.int16(v) != nodata_i16:
                if has_v == 0:
                    has_v = np.uint8(1)
                    lo = v
                    hi = v
                else:
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        row_has[r] = has_v
        row_min[r] = lo
        row_max[r] = hi
    has_any = False
    lo = 255
    hi = 0
    for r in range(h):
        if row_has[r] != 0:
            v_lo = int(row_min[r])
            v_hi = int(row_max[r])
            if not has_any:
                has_any = True
                lo = v_lo
                hi = v_hi
            else:
                if v_lo < lo:
                    lo = v_lo
                if v_hi > hi:
                    hi = v_hi
    return has_any, lo, hi


@njit(parallel=True)
def _apply_lut_u8_2d(src, lut, dst):
    h, w = src.shape
    for r in prange(h):
        for c in range(w):
            dst[r, c] = lut[src[r, c]]


@njit(parallel=True)
def _apply_lut_u8_inplace(arr_u8, lut_u8):
    h, w = arr_u8.shape
    for r in prange(h):
        for c in range(w):
            arr_u8[r, c] = lut_u8[arr_u8[r, c]]


@njit(parallel=True)
def _encode_outputs_2d(chunk_gc, chunk_ref):
    """Compute uint8 DGC and scaled reference cover in one parallel pass."""
    h, w = chunk_ref.shape
    dgc_data = np.empty((h, w), dtype=np.uint8)
    ref_data_scaled = np.empty((h, w), dtype=np.uint8)
    for r in prange(h):
        for c in range(w):
            rv = chunk_ref[r, c]
            if rv <= 0 or rv > 101:
                ref_data_scaled[r, c] = 0
                dgc_data[r, c] = 0
            else:
                ref_data_scaled[r, c] = np.uint8(100 + rv)
                gv = chunk_gc[r, c]
                if gv <= 0 or gv > 101:
                    dgc_data[r, c] = 0
                else:
                    diff = 100 + int(gv) - int(rv)
                    if diff < 0:
                        diff = 0
                    elif diff > 255:
                        diff = 255
                    dgc_data[r, c] = np.uint8(diff)
    return dgc_data, ref_data_scaled


def _infer_encoding_from_array(bare_data, nodata_val=255):
    """
    Infer bare ground encoding ('dp8a2' vs 'dixa2') and whether evidence is decisive
    from valid values in [0, 200] excluding nodata.
    """
    nd_int = None
    if nodata_val is not None:
        try:
            cand = int(nodata_val)
            if cand == nodata_val:
                nd_int = cand
        except (TypeError, ValueError, OverflowError):
            pass

    if bare_data.dtype == np.uint8 and bare_data.ndim == 2:
        nd_i16 = np.int16(nd_int) if (nd_int is not None and 0 <= nd_int <= 255) else np.int16(-1)
        has_valid, min_valid, max_valid = _scan_valid_minmax_u8_2d(bare_data, nd_i16)
    else:
        valid_cand = (bare_data >= 0) & (bare_data <= 200)
        if nodata_val is not None:
            valid_cand &= bare_data != nodata_val
        has_valid = bool(np.any(valid_cand))
        if has_valid:
            vals = bare_data[valid_cand]
            min_valid = int(np.min(vals))
            max_valid = int(np.max(vals))
        else:
            min_valid = 255
            max_valid = 0

    if has_valid and max_valid > 100:
        return "dixa2", True
    if has_valid and min_valid < 100:
        decisive = (min_valid > 0) or (nd_int != 0)
        return "dp8a2", decisive
    return ("dixa2" if nd_int == 0 else "dp8a2"), False


def convert_to_groundcover(bare_data, nodata_val=255, inplace=False, encoding=None):
    """
    Convert raw bare ground array to persistent groundcover + 1 (values 1..101, 0 is nodata).

    Supports both remote sensing conventions:
    - Modern DEA / RSC dp8a2: Bare stored as 0..100% with 255 NoData -> 101 - bare (1..101).
    - Legacy JRSRP / RSC dixa2: Bare stored with +100 offset (100..200) -> 201 - bare (1..101).
    If encoding ('dp8a2' or 'dixa2') is specified, applies the 256-entry LUT in one pass.
    """
    if bare_data.size == 0:
        return np.empty(bare_data.shape, dtype=np.int8)

    if encoding is None:
        encoding, _ = _infer_encoding_from_array(bare_data, nodata_val)
    elif encoding not in ("dp8a2", "dixa2"):
        raise ValueError(f"Unsupported bare ground encoding: {encoding}")

    is_dixa2 = encoding == "dixa2"
    nd_int = None
    if nodata_val is not None:
        try:
            cand = int(nodata_val)
            if cand == nodata_val:
                nd_int = cand
        except (TypeError, ValueError, OverflowError):
            pass

    if bare_data.dtype == np.uint8 and bare_data.ndim == 2:
        lut = np.zeros(256, dtype=np.int8)
        if is_dixa2:
            lut[100:201] = np.arange(101, 0, -1, dtype=np.int8)
        else:
            lut[0:101] = np.arange(101, 0, -1, dtype=np.int8)
        if nd_int is not None and 0 <= nd_int <= 255:
            lut[nd_int] = 0
        if inplace and bare_data.flags.writeable:
            # Output range 0..101 is bit-identical across uint8 and int8 views.
            _apply_lut_u8_inplace(bare_data, lut.view(np.uint8))
            return bare_data.view(np.int8)
        out = np.empty(bare_data.shape, dtype=np.int8)
        _apply_lut_u8_2d(bare_data, lut, out)
        return out

    if is_dixa2:
        valid = (bare_data >= 100) & (bare_data <= 200)
        top_off = 201
    else:
        valid = (bare_data >= 0) & (bare_data <= 100)
        top_off = 101
    if nodata_val is not None:
        valid &= bare_data != nodata_val
    gc = np.where(valid, top_off - bare_data.astype(np.int32), 0)
    return np.clip(gc, 0, 101).astype(np.int8)


def _get_effective_cpus():
    """Detect effective CPU count honoring Linux container cgroup v1/v2 quotas."""
    host_cpus = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            if quota > 0 and period > 0:
                return min(host_cpus, max(1, int(round(quota / period))))
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota = int(fq.read().strip())
            period = int(fp.read().strip())
        if quota > 0 and period > 0:
            return min(host_cpus, max(1, int(round(quota / period))))
    except OSError:
        pass
    return host_cpus


def _read_clipped_band_window(read_fn, band_w, band_h, win, fill=0):
    """
    Read a spatial window from a band, using a direct read when completely inside
    [0, band_w) x [0, band_h) and padding any out-of-bounds margins with ``fill``.
    """
    xoff, yoff, xsize, ysize = map(int, win)
    if xsize <= 0 or ysize <= 0:
        return np.empty((max(0, ysize), max(0, xsize)), dtype=np.uint8)
    if xoff >= 0 and yoff >= 0 and (xoff + xsize) <= band_w and (yoff + ysize) <= band_h:
        return read_fn((xoff, yoff, xsize, ysize))

    ix0 = max(0, xoff)
    iy0 = max(0, yoff)
    ix1 = min(band_w, xoff + xsize)
    iy1 = min(band_h, yoff + ysize)
    if ix1 > ix0 and iy1 > iy0:
        sub = read_fn((ix0, iy0, ix1 - ix0, iy1 - iy0))
        out = np.full((ysize, xsize), fill, dtype=sub.dtype)
        out[(iy0 - yoff):(iy1 - yoff), (ix0 - xoff):(ix1 - xoff)] = sub
        return out
    return np.full((ysize, xsize), fill, dtype=np.uint8)


class _LocalBandReader:
    """
    Read-only local GDAL band reader used exclusively from a single dedicated
    I/O prefetch thread while the main thread runs Numba compute and writes output.
    """

    def __init__(self, in_path, ref_path, ibra_path=None, ibra_band_idx=None):
        self.in_ds = gdal.Open(in_path, gdal.GA_ReadOnly)
        self.in_band = self.in_ds.GetRasterBand(1)
        self.ref_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
        self.ref_band = self.ref_ds.GetRasterBand(1)
        self.ref_w, self.ref_h = self.ref_ds.RasterXSize, self.ref_ds.RasterYSize

        self.ibra_ds = None
        self.ibra_band = None
        self.ibra_w, self.ibra_h = self.ref_w, self.ref_h
        if ibra_band_idx is not None:
            if ibra_path:
                self.ibra_ds = gdal.Open(ibra_path, gdal.GA_ReadOnly)
                self.ibra_band = self.ibra_ds.GetRasterBand(ibra_band_idx)
                self.ibra_w, self.ibra_h = self.ibra_ds.RasterXSize, self.ibra_ds.RasterYSize
            else:
                self.ibra_band = self.ref_ds.GetRasterBand(ibra_band_idx)

    def read_all_windows(self, in_win, ref_win, need_ibra=False, ibra_win=None):
        """Sequentially read aligned input, reference, and optional IBRA windows."""
        bare = self.in_band.ReadAsArray(*map(int, in_win))

        perc_u8 = _read_clipped_band_window(
            lambda w: self.ref_band.ReadAsArray(*w), self.ref_w, self.ref_h, ref_win, fill=0
        )
        perc = perc_u8.view(np.int8) if perc_u8.dtype == np.uint8 else perc_u8.astype(np.int8)

        ibra = None
        if need_ibra:
            eff_ib_win = ref_win if ibra_win is None else ibra_win
            ib_arr = _read_clipped_band_window(
                lambda w: self.ibra_band.ReadAsArray(*w), self.ibra_w, self.ibra_h, eff_ib_win, fill=0
            )
            ibra = ib_arr if ib_arr.dtype == np.uint8 else ib_arr.astype(np.uint8)

        return bare, perc, ibra

    def close(self):
        self.in_band = None
        self.in_ds = None
        self.ref_band = None
        self.ref_ds = None
        self.ibra_band = None
        self.ibra_ds = None


def _resolve_bbox_bounds(bbox=None, geojson=None, target_proj_wkt=None):
    """Resolve projected AOI bounds (minx, miny, maxx, maxy) in the target raster CRS."""
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"Expected 4 bbox values [minx, miny, maxx, maxy], got {bbox}")
        minx, miny, maxx, maxy = map(float, bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"Invalid bbox bounds: {(minx, miny, maxx, maxy)}")
        return minx, miny, maxx, maxy

    ds = ogr.Open(geojson)
    if ds is None:
        raise ValueError(f"Could not open vector AOI: {geojson}")
    try:
        lyr = ds.GetLayer(0)
        src_srs = lyr.GetSpatialRef()

        transform = None
        if target_proj_wkt:
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(target_proj_wkt)
            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if src_srs is not None:
                src_srs_norm = src_srs.Clone()
                src_srs_norm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                if not src_srs_norm.IsSame(dst_srs):
                    transform = osr.CoordinateTransformation(src_srs_norm, dst_srs)
            else:
                crs84 = osr.SpatialReference()
                crs84.ImportFromEPSG(4326)
                crs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                if not crs84.IsSame(dst_srs):
                    transform = osr.CoordinateTransformation(crs84, dst_srs)

        if transform is None:
            ext = lyr.GetExtent(force=1)
            return float(ext[0]), float(ext[2]), float(ext[1]), float(ext[3])

        minx = math.inf
        miny = math.inf
        maxx = -math.inf
        maxy = -math.inf
        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            g = geom.Clone()
            g.Transform(transform)
            gx0, gx1, gy0, gy1 = g.GetEnvelope()
            minx = min(minx, gx0)
            maxx = max(maxx, gx1)
            miny = min(miny, gy0)
            maxy = max(maxy, gy1)
        if not math.isfinite(minx):
            raise ValueError(f"No valid geometries found in vector AOI: {geojson}")
        return minx, miny, maxx, maxy
    finally:
        ds = None


def _validate_aligned_grid(in_gt, in_proj, target_gt, target_proj, target_label):
    """
    Validate axis-aligned grid compatibility, pixel resolution, CRS identity, and
    integer pixel offset (dx, dy) between input_groundcover and reference/IBRA raster.
    """
    if in_gt[2] != 0 or in_gt[4] != 0 or in_gt[1] <= 0 or in_gt[5] >= 0:
        raise ValueError(f"Expected north-up axis-aligned GeoTransform for input raster, got {in_gt}")
    if target_gt[2] != 0 or target_gt[4] != 0 or target_gt[1] <= 0 or target_gt[5] >= 0:
        raise ValueError(f"Expected north-up axis-aligned GeoTransform for {target_label}, got {target_gt}")
    if not (
        math.isclose(in_gt[1], target_gt[1], rel_tol=1e-6, abs_tol=1e-9)
        and math.isclose(in_gt[5], target_gt[5], rel_tol=1e-6, abs_tol=1e-9)
    ):
        raise ValueError(
            f"Pixel size mismatch between input ({in_gt[1]}, {in_gt[5]}) and {target_label} ({target_gt[1]}, {target_gt[5]})"
        )

    if in_proj and target_proj:
        s_in = osr.SpatialReference()
        s_in.ImportFromWkt(in_proj)
        s_in.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        s_tgt = osr.SpatialReference()
        s_tgt.ImportFromWkt(target_proj)
        s_tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not s_in.IsSame(s_tgt):
            raise ValueError(f"Projection CRS mismatch between input raster and {target_label}.")

    dx_f = (in_gt[0] - target_gt[0]) / target_gt[1]
    dy_f = (in_gt[3] - target_gt[3]) / target_gt[5]
    dx = int(round(dx_f))
    dy = int(round(dy_f))
    if abs(dx_f - dx) > 1e-3 or abs(dy_f - dy) > 1e-3:
        raise ValueError(
            f"Sub-pixel grid misalignment between input and {target_label}: dx={dx_f:.6f}, dy={dy_f:.6f}"
        )
    return dx, dy


def _chunk_bounds(chunk, chunksize, full_height, y_origin_off, in_h, ref_h, in_to_ref_dy, halfwindow, pad_halo):
    """Compute exact source/reference windows and output row slices for a chunk."""
    datastartline = chunk * chunksize
    datastopline = min((chunk + 1) * chunksize, full_height)
    if pad_halo:
        in_read_y0 = max(0, y_origin_off + datastartline - halfwindow)
        in_read_y1 = min(in_h, y_origin_off + datastopline + halfwindow)
    else:
        in_read_y0 = y_origin_off + max(0, datastartline - halfwindow)
        in_read_y1 = y_origin_off + min(full_height, datastopline + halfwindow)
    ref_read_y = in_to_ref_dy + in_read_y0
    y_slice_start = (y_origin_off + datastartline) - in_read_y0
    y_slice_stop = (y_origin_off + datastopline) - in_read_y0
    return datastartline, datastopline, in_read_y0, in_read_y1, ref_read_y, y_slice_start, y_slice_stop


def run_seasonal_drcm_ibra(
    input_groundcover,
    percentile_file,
    ibra_raster=None,
    output_file=None,
    window=1501,
    chunksize=4096,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    test_chip=None,
    bbox=None,
    geojson=None,
    threads=None,
    overview_levels=(4, 8, 16, 32),
    no_overviews=False,
    no_ibra=False,
    bare_encoding="auto",
    min_obs_percent=0.0,
    sparse_min_valpercent=0.0,
    min_ref_samples=0,
):
    """
    Execute IBRA-constrained (or unconstrained ``no_ibra=True``) seasonal DRCM processing.

    Produces a 2-band uint8 GeoTIFF:
      - Band 1: Difference Ground Cover (``100 + gc - ref``, ``0 = NoData``)
      - Band 2: Reference Ground Cover (``100 + ref``, ``0 = NoData``)
    """
    eff_cpus = threads if threads is not None else _get_effective_cpus()
    if threads is not None or os.environ.get("NUMBA_NUM_THREADS") is None:
        import numba
        eff_cpus = max(1, int(eff_cpus))
        numba.set_num_threads(eff_cpus)

    if window % 2 == 0:
        window += 1

    n_spatial_modes = sum(x is not None for x in (test_chip, bbox, geojson))
    if n_spatial_modes > 1:
        raise ValueError("Specify at most one of test_chip, bbox, or geojson.")

    if output_file is None:
        output_file = derive_output_name(input_groundcover, no_ibra=no_ibra)
    elif isinstance(output_file, str) and output_file.endswith(("/", os.sep)):
        output_file = os.path.join(output_file, derive_output_name(input_groundcover, no_ibra=no_ibra))

    mode_str = "Single Global Histogram (IBRA disabled)" if no_ibra else "IBRA-Constrained Regional Histograms"
    print(
        f"Processing Mode: {mode_str} | effective_cpus={eff_cpus} | "
        f"gates(minval={minvalpercent}, min_obs={min_obs_percent}, "
        f"sparse_min_val={sparse_min_valpercent}, min_ref_samples={min_ref_samples})",
        flush=True,
    )

    in_ds = ref_ds = ibra_ds_tmp = None
    try:
        print(f"Opening input groundcover: {input_groundcover}", flush=True)
        in_ds = gdal.Open(input_groundcover, gdal.GA_ReadOnly)
        in_gt = in_ds.GetGeoTransform()
        in_proj = in_ds.GetProjection()
        in_band = in_ds.GetRasterBand(1)
        nodata_in = in_band.GetNoDataValue()
        if nodata_in is None:
            nodata_in = 255
        in_w, in_h = in_ds.RasterXSize, in_ds.RasterYSize

        print(f"Opening reference stats file: {percentile_file}", flush=True)
        ref_ds = gdal.Open(percentile_file, gdal.GA_ReadOnly)
        ref_gt = ref_ds.GetGeoTransform()
        ref_proj = ref_ds.GetProjection()
        ref_h = ref_ds.RasterYSize

        ibra_band_idx = None
        ibra_path_eff = ibra_raster
        ibra_gt = ref_gt
        ibra_proj = ref_proj
        if not no_ibra:
            if ref_ds.RasterCount >= 2:
                ibra_band_idx = 2
                ibra_path_eff = None
            elif ibra_path_eff:
                print(f"Opening separate IBRA raster: {ibra_raster}", flush=True)
                ibra_ds_tmp = gdal.Open(ibra_path_eff, gdal.GA_ReadOnly)
                if ibra_ds_tmp.RasterCount < 1:
                    raise ValueError(f"IBRA raster has no bands: {ibra_raster}")
                ibra_gt = ibra_ds_tmp.GetGeoTransform()
                ibra_proj = ibra_ds_tmp.GetProjection()
                ibra_band_idx = 1
            else:
                raise ValueError(
                    f"{percentile_file} has only 1 band. Specify --ibra-raster, provide a 2-band masked stats+IBRA raster, or use --no-ibra."
                )
    finally:
        ibra_ds_tmp = None
        in_band = None
        in_ds = None
        ref_ds = None

    if test_chip:
        x_origin_off, y_origin_off, full_width, full_height = test_chip
        if (
            x_origin_off < 0
            or y_origin_off < 0
            or full_width <= 0
            or full_height <= 0
            or (x_origin_off + full_width) > in_w
            or (y_origin_off + full_height) > in_h
        ):
            raise ValueError(f"--test-chip {test_chip} out of bounds for input raster {in_w}x{in_h}")
        pad_halo = False
        print(f"Running on test chip: xoff={x_origin_off}, yoff={y_origin_off}, w={full_width}, h={full_height}", flush=True)
    elif bbox is not None or geojson is not None:
        minx, miny, maxx, maxy = _resolve_bbox_bounds(bbox=bbox, geojson=geojson, target_proj_wkt=in_proj)
        if in_gt[2] != 0 or in_gt[4] != 0 or in_gt[1] <= 0 or in_gt[5] >= 0:
            raise ValueError(f"Expected north-up axis-aligned GeoTransform for AOI subsetting, got {in_gt}")
        x_origin_off = max(0, min(in_w, math.floor((minx - in_gt[0]) / in_gt[1])))
        x_end = max(0, min(in_w, math.ceil((maxx - in_gt[0]) / in_gt[1])))
        y_origin_off = max(0, min(in_h, math.floor((maxy - in_gt[3]) / in_gt[5])))
        y_end = max(0, min(in_h, math.ceil((miny - in_gt[3]) / in_gt[5])))
        full_width = x_end - x_origin_off
        full_height = y_end - y_origin_off
        if full_width <= 0 or full_height <= 0:
            raise ValueError(f"AOI does not intersect input raster: x=({x_origin_off}, {x_end}), y=({y_origin_off}, {y_end})")
        pad_halo = True
        print(
            f"Subsetting AOI ({bbox if bbox is not None else geojson}) -> "
            f"xoff={x_origin_off}, yoff={y_origin_off}, w={full_width}, h={full_height} (halo-padded)",
            flush=True,
        )
    else:
        x_origin_off, y_origin_off = 0, 0
        full_width = in_w
        full_height = in_h
        pad_halo = False

    chip_gt = (
        in_gt[0] + x_origin_off * in_gt[1],
        in_gt[1],
        in_gt[2],
        in_gt[3] + y_origin_off * in_gt[5],
        in_gt[4],
        in_gt[5],
    )

    in_to_ref_dx, in_to_ref_dy = _validate_aligned_grid(in_gt, in_proj, ref_gt, ref_proj, "reference raster")
    if not no_ibra and ibra_path_eff is not None:
        in_to_ibra_dx, in_to_ibra_dy = _validate_aligned_grid(in_gt, in_proj, ibra_gt, ibra_proj, "IBRA raster")
    else:
        in_to_ibra_dx, in_to_ibra_dy = in_to_ref_dx, in_to_ref_dy

    if bare_encoding in ("dp8a2", "dixa2"):
        locked_encoding = bare_encoding
        print(f"Bare ground encoding: {locked_encoding} (explicit)", flush=True)
    elif bare_encoding == "auto":
        locked_encoding = infer_bare_encoding_from_filename(input_groundcover)
        if locked_encoding is not None:
            print(f"Bare ground encoding: {locked_encoding} (from filename stage)", flush=True)
    else:
        raise ValueError(f"Unsupported bare_encoding: {bare_encoding}")

    print(
        f"Spatial alignment offset in reference grid: col={in_to_ref_dx + x_origin_off}, "
        f"row={in_to_ref_dy + y_origin_off}",
        flush=True,
    )

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    halfwindow = window // 2
    if pad_halo:
        in_read_x0 = max(0, x_origin_off - halfwindow)
        in_read_x1 = min(in_w, x_origin_off + full_width + halfwindow)
    else:
        in_read_x0 = x_origin_off
        in_read_x1 = x_origin_off + full_width
    win_xsize = in_read_x1 - in_read_x0
    x_slice_start = x_origin_off - in_read_x0
    x_slice_stop = x_slice_start + full_width

    ref_read_x = in_to_ref_dx + in_read_x0
    ibra_read_x = in_to_ibra_dx + in_read_x0

    reader = _LocalBandReader(
        input_groundcover,
        percentile_file,
        ibra_path=ibra_path_eff,
        ibra_band_idx=ibra_band_idx,
    )

    t_start = time.perf_counter()
    total_pixels_processed = 0
    total_io_wait_s = 0.0
    total_io_work_s = 0.0
    total_compute_s = 0.0
    total_write_s = 0.0
    t_ovr = 0.0

    def _plan_chunk_fetch(chunk_idx, prev_meta):
        ds, de, y0, y1, ry, ys0, ys1 = _chunk_bounds(
            chunk_idx, chunksize, full_height, y_origin_off, in_h, ref_h, in_to_ref_dy, halfwindow, pad_halo
        )
        wys = y1 - y0
        reuse_ok = (
            prev_meta is not None
            and prev_meta[1] > y0
            and prev_meta[0] <= y0
            and (ry - y0) == (prev_meta[2] - prev_meta[0])
        )
        if reuse_ok:
            ov_y1 = min(y1, prev_meta[1])
            ov_rows = ov_y1 - y0
            tail_h = y1 - ov_y1
            fetch_spec = (ov_y1, ry + ov_rows, tail_h)
        else:
            ov_rows = 0
            fetch_spec = (y0, ry, wys)
        return {
            "datastartline": ds,
            "datastopline": de,
            "y0": y0,
            "y1": y1,
            "ref_y": ry,
            "win_ysize": wys,
            "y_slice_start": ys0,
            "y_slice_stop": ys1,
            "reuse_ok": reuse_ok,
            "ov_rows": ov_rows,
            "fetch_spec": fetch_spec,
        }

    def _fetch_tail(fetch_spec):
        read_y_in, read_y_ref, read_h = fetch_spec
        t0 = time.perf_counter()
        if read_h <= 0:
            return None, None, None, 0.0
        ibra_win = (ibra_read_x, in_to_ibra_dy + read_y_in, win_xsize, read_h) if not no_ibra else None
        bare, perc_part, ibra_part = reader.read_all_windows(
            (in_read_x0, read_y_in, win_xsize, read_h),
            (ref_read_x, read_y_ref, win_xsize, read_h),
            need_ibra=(not no_ibra),
            ibra_win=ibra_win,
        )
        return bare, perc_part, ibra_part, (time.perf_counter() - t0)

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    out_ds = dgc_band = ref_out_band = None
    try:
        print(f"Creating output dataset: {output_file} ({full_width} x {full_height})", flush=True)
        driver = gdal.GetDriverByName("GTiff")
        options = [
            "COMPRESS=DEFLATE",
            "ZLEVEL=1",
            "PREDICTOR=2",
            "TILED=YES",
            "INTERLEAVE=BAND",
            "BIGTIFF=YES",
            f"NUM_THREADS={eff_cpus}",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ]
        out_ds = driver.Create(output_file, full_width, full_height, 2, gdal.GDT_Byte, options=options)
        out_ds.SetGeoTransform(chip_gt)
        out_ds.SetProjection(in_proj)

        dgc_band = out_ds.GetRasterBand(1)
        dgc_desc = (
            "Difference Ground Cover (Unconstrained, 100+gc-ref, 0=NoData)"
            if no_ibra
            else "Difference Ground Cover (IBRA-Constrained, 100+gc-ref, 0=NoData)"
        )
        dgc_band.SetDescription(dgc_desc)
        dgc_band.SetNoDataValue(0)

        ref_out_band = out_ds.GetRasterBand(2)
        ref_desc = (
            "Reference Ground Cover (Unconstrained, 100+ref, 0=NoData)"
            if no_ibra
            else "Reference Ground Cover (IBRA-Constrained, 100+ref, 0=NoData)"
        )
        ref_out_band.SetDescription(ref_desc)
        ref_out_band.SetNoDataValue(0)

        nchunks = (full_height + chunksize - 1) // chunksize
        print(f"Processing {full_height} lines in {nchunks} chunks (chunksize={chunksize}, window={window})...\n", flush=True)

        prev_state = None
        cur_plan = _plan_chunk_fetch(0, None)
        next_future = prefetch_pool.submit(_fetch_tail, cur_plan["fetch_spec"])

        for chunk in range(nchunks):
            t_chunk_start = time.perf_counter()
            datastartline = cur_plan["datastartline"]
            datastopline = cur_plan["datastopline"]
            chunk_height = datastopline - datastartline
            in_read_y0 = cur_plan["y0"]
            in_read_y1 = cur_plan["y1"]
            ref_read_y = cur_plan["ref_y"]
            win_ysize = cur_plan["win_ysize"]
            y_slice_start = cur_plan["y_slice_start"]
            y_slice_stop = cur_plan["y_slice_stop"]

            pct_start = (datastartline / full_height) * 100
            print(f"[Chunk {chunk + 1}/{nchunks} - {pct_start:5.1f}%] Starting lines {datastartline}:{datastopline}...", flush=True)

            t_io0 = time.perf_counter()
            bare_tail, perc_tail, ibra_tail, io_work_s = next_future.result()
            total_io_work_s += io_work_s
            if bare_tail is not None:
                if locked_encoding is None:
                    # Only reached for non-standard filenames lacking dp8/dix stage tokens.
                    chunk_enc, is_decisive = _infer_encoding_from_array(bare_tail, nodata_in)
                    if is_decisive:
                        locked_encoding = chunk_enc
                else:
                    chunk_enc = locked_encoding
                gc_tail = convert_to_groundcover(bare_tail, nodata_in, inplace=True, encoding=chunk_enc)
            else:
                gc_tail = None
            del bare_tail

            if cur_plan["reuse_ok"]:
                ov_rows = cur_plan["ov_rows"]
                prev_s = in_read_y0 - prev_state["y0"]
                prev_e = prev_s + ov_rows

                # Pop prev_state arrays incrementally so old chunk buffers are freed immediately.
                groundcover_data = np.empty((win_ysize, win_xsize), dtype=np.int8)
                prev_gc = prev_state.pop("gc")
                groundcover_data[:ov_rows] = prev_gc[prev_s:prev_e]
                del prev_gc
                if gc_tail is not None:
                    groundcover_data[ov_rows:] = gc_tail
                gc_tail = None

                percentile_data = np.empty((win_ysize, win_xsize), dtype=np.int8)
                prev_perc = prev_state.pop("perc")
                percentile_data[:ov_rows] = prev_perc[prev_s:prev_e]
                del prev_perc
                if perc_tail is not None:
                    percentile_data[ov_rows:] = perc_tail
                perc_tail = None

                if not no_ibra:
                    ibra_data = np.empty((win_ysize, win_xsize), dtype=np.uint8)
                    prev_ib = prev_state.pop("ibra")
                    ibra_data[:ov_rows] = prev_ib[prev_s:prev_e]
                    del prev_ib
                    if ibra_tail is not None:
                        ibra_data[ov_rows:] = ibra_tail
                    ibra_tail = None
                else:
                    ibra_data = None
                prev_state = None
            else:
                prev_state = None
                groundcover_data = gc_tail
                percentile_data = perc_tail
                ibra_data = ibra_tail
                gc_tail = perc_tail = ibra_tail = None
            t_io1 = time.perf_counter()
            io_wait_s = t_io1 - t_io0
            total_io_wait_s += io_wait_s

            # Enqueue next chunk I/O before entering Numba compute + disk write.
            if chunk + 1 < nchunks:
                next_plan = _plan_chunk_fetch(chunk + 1, (in_read_y0, in_read_y1, ref_read_y))
                next_future = prefetch_pool.submit(_fetch_tail, next_plan["fetch_spec"])
                cur_plan = next_plan
            else:
                next_future = None

            t_cmp0 = time.perf_counter()
            if not no_ibra:
                chunk_ref = refpixelfilter_ibra(
                    percentile_data,
                    groundcover_data,
                    ibra_data,
                    window=window,
                    lowerquantile=lowerquantile,
                    upperquantile=upperquantile,
                    minvalpercent=minvalpercent,
                    r_start=y_slice_start,
                    r_stop=y_slice_stop,
                    c_start=x_slice_start,
                    c_stop=x_slice_stop,
                    min_obs_percent=min_obs_percent,
                    sparse_min_valpercent=sparse_min_valpercent,
                    min_ref_samples=min_ref_samples,
                )
            else:
                chunk_ref = refpixelfilternodata(
                    percentile_data,
                    groundcover_data,
                    window=window,
                    lowerquantile=lowerquantile,
                    upperquantile=upperquantile,
                    minvalpercent=minvalpercent,
                    r_start=y_slice_start,
                    r_stop=y_slice_stop,
                    c_start=x_slice_start,
                    c_stop=x_slice_stop,
                    min_obs_percent=min_obs_percent,
                    sparse_min_valpercent=sparse_min_valpercent,
                    min_ref_samples=min_ref_samples,
                )

            if chunk + 1 >= nchunks:
                del percentile_data, ibra_data

            chunk_gc = groundcover_data[y_slice_start:y_slice_stop, x_slice_start:x_slice_stop]
            dgc_data, ref_data_scaled = _encode_outputs_2d(chunk_gc, chunk_ref)
            del chunk_ref
            t_cmp1 = time.perf_counter()
            total_compute_s += t_cmp1 - t_cmp0

            if chunk + 1 < nchunks:
                prev_state = {
                    "y0": in_read_y0,
                    "y1": in_read_y1,
                    "ref_y0": ref_read_y,
                    "gc": groundcover_data,
                    "perc": percentile_data,
                    "ibra": ibra_data,
                }
            else:
                del groundcover_data

            t_wr0 = time.perf_counter()
            dgc_band.WriteArray(dgc_data, 0, datastartline)
            ref_out_band.WriteArray(ref_data_scaled, 0, datastartline)
            del dgc_data, ref_data_scaled
            dgc_band.FlushCache()
            ref_out_band.FlushCache()
            t_wr1 = time.perf_counter()
            total_write_s += t_wr1 - t_wr0

            chunk_pixels = chunk_height * full_width
            total_pixels_processed += chunk_pixels
            t_chunk = time.perf_counter() - t_chunk_start
            t_elapsed = time.perf_counter() - t_start
            rate = chunk_pixels / t_chunk if t_chunk > 0 else 0
            overall_rate = total_pixels_processed / t_elapsed if t_elapsed > 0 else 0

            pct_done = (datastopline / full_height) * 100
            remaining_pixels = (full_height - datastopline) * full_width
            eta_seconds = remaining_pixels / overall_rate if overall_rate > 0 else 0

            print(
                f"[Chunk {chunk + 1}/{nchunks} - {pct_done:5.1f}%] Lines {datastartline}:{datastopline} done in {t_chunk:.1f}s "
                f"(io_wait={io_wait_s:.1f}s, io_bg={io_work_s:.1f}s, numba={t_cmp1-t_cmp0:.1f}s, write={t_wr1-t_wr0:.1f}s, {rate/1e6:.2f}M px/s) | "
                f"Elapsed: {format_time(t_elapsed)} | Remaining: ~{format_time(eta_seconds)}\n",
                flush=True,
            )

        out_ds.FlushCache()

        if not no_overviews and overview_levels:
            ovr_str = ", ".join(str(lvl) for lvl in overview_levels)
            print(f"Calculating internal overviews (average, levels: {ovr_str})...", flush=True)
            t_ovr_start = time.perf_counter()
            prev_cfg = {
                key: gdal.GetConfigOption(key)
                for key in ("COMPRESS_OVERVIEW", "PREDICTOR_OVERVIEW", "NUM_THREADS", "GDAL_NUM_THREADS")
            }
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            gdal.SetConfigOption("PREDICTOR_OVERVIEW", "1")
            gdal.SetConfigOption("NUM_THREADS", str(eff_cpus))
            gdal.SetConfigOption("GDAL_NUM_THREADS", str(eff_cpus))
            try:
                out_ds.BuildOverviews("AVERAGE", list(overview_levels), callback=gdal.TermProgress_nocb)
                out_ds.FlushCache()
            finally:
                for key, val in prev_cfg.items():
                    gdal.SetConfigOption(key, val)
            t_ovr = time.perf_counter() - t_ovr_start
            print(f"\nOverviews completed in {t_ovr:.1f}s", flush=True)

        dgc_band = None
        ref_out_band = None
        out_ds = None

        t_total = time.perf_counter() - t_start
        overall_rate = total_pixels_processed / t_total if t_total > 0 else 0
        if resource is not None:
            divisor = 1048576.0 if sys.platform == "darwin" else 1024.0
            rss_str = f", max_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor:.1f}MB"
        else:
            rss_str = ""
        print(
            f"\n[100.0%] Finished generating {output_file}\n"
            f"Total: {total_pixels_processed:,} pixels in {format_time(t_total)} "
            f"(wall={t_total:.2f}s | avg {overall_rate/1e6:.2f}M px/s | "
            f"io_wait={total_io_wait_s:.1f}s, io_bg={total_io_work_s:.1f}s, "
            f"numba={total_compute_s:.1f}s, write={total_write_s:.1f}s, ovr={t_ovr:.1f}s{rss_str})",
            flush=True,
        )
        return output_file
    finally:
        dgc_band = None
        ref_out_band = None
        out_ds = None
        prefetch_pool.shutdown(wait=True)
        reader.close()


def main(argv=None):
    args = parse_args(argv)
    quantile_span = max(0.0, args.upperquantile - args.lowerquantile)
    max_quantile_samples = int(args.window * args.window * quantile_span)
    if args.min_ref_samples > 0 and args.min_ref_samples > max_quantile_samples:
        print(
            f"WARNING: --min-ref-samples={args.min_ref_samples} exceeds max quantile sample count "
            f"(~{max_quantile_samples} for window={args.window}, [{args.lowerquantile}, {args.upperquantile}]); "
            "pass --min-ref-samples 0 for small windows.",
            flush=True,
        )
    run_seasonal_drcm_ibra(
        input_groundcover=args.input_groundcover,
        percentile_file=args.percentile_file,
        ibra_raster=args.ibra_raster,
        output_file=args.output,
        window=args.window,
        chunksize=args.chunksize,
        lowerquantile=args.lowerquantile,
        upperquantile=args.upperquantile,
        minvalpercent=args.minvalpercent,
        test_chip=args.test_chip,
        bbox=args.bbox,
        geojson=args.geojson,
        threads=args.threads,
        overview_levels=args.overviews,
        no_overviews=args.no_overviews,
        no_ibra=args.no_ibra,
        bare_encoding=args.bare_encoding,
        min_obs_percent=args.min_obs_percent,
        sparse_min_valpercent=args.sparse_min_valpercent,
        min_ref_samples=args.min_ref_samples,
    )


if __name__ == "__main__":
    main()
