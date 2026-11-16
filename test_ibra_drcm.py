"""
test_ibra_drcm.py

Verification test suite for the Dynamic Reference Cover Method (DRCM) pipeline.
Covers an independent histogram-free mathematical oracle, Serpentine 2D tiled
Numba kernels, GeoTIFF format/predictor/interleave handling, chunk prefetch &
overlap reuse, halo-padded AOI subsetting, Stage-1 preprocessing, sparse-reference
cloud recovery, and portable real-satellite GeoTIFF chips in ``test_data/``.

Run the full suite:
  python test_ibra_drcm.py
"""

import os
import sys
import tempfile
import time

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import maskChangeSteepIbra
import refpix_numba
import refpix_numba_ibra
import seasonalDRCMIbra
from seasonalDRCMIbra import (
    _LocalBandReader,
    convert_to_groundcover,
    derive_output_name,
    format_time,
    infer_bare_encoding_from_filename,
    run_seasonal_drcm_ibra,
)

TEST_DATA_DIR = os.path.join(REPO_ROOT, "test_data")
LOCAL_REF_CHIP = os.path.join(TEST_DATA_DIR, "lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif")
LOCAL_GC_CHIP = os.path.join(TEST_DATA_DIR, "lztmre_aus_m202606202608_dp8a2_vinterim.tif")
LOCAL_AOI_GEOJSON = os.path.join(TEST_DATA_DIR, "aoi_nsw_ibra_boundary.geojson")


def _oracle_window_passes(
    n_dom, n_pop, n_valid, minvalpercent, min_obs_percent, sparse_min_valpercent
):
    if n_valid <= 0:
        return False
    if n_valid >= float(np.float32(minvalpercent)) * n_dom:
        return True
    min_obs_f = float(np.float32(min_obs_percent))
    sparse_f = float(np.float32(sparse_min_valpercent))
    if min_obs_f > 0.0 and n_pop > 0:
        if n_valid >= min_obs_f * n_pop and n_pop >= sparse_f * n_dom:
            return True
    return False


def _oracle_refpixel(
    cover,
    ground,
    ibra=None,
    window=15,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    r_slice=None,
    c_slice=None,
    min_obs_percent=0.0,
    sparse_min_valpercent=0.0,
    min_ref_samples=0,
):
    """
    Independent histogram-free brute-force oracle built directly from 2D numpy slices
    and sorted value arrays (shares zero code with refpix_numba / refpix_numba_ibra).
    """
    nrow, ncol = cover.shape
    hw = window // 2
    r0_out, r1_out = (0, nrow) if r_slice is None else r_slice
    c0_out, c1_out = (0, ncol) if c_slice is None else c_slice
    out = np.empty((r1_out - r0_out, c1_out - c0_out), dtype=np.int8)
    eff_min_samples = max(0, int(min_ref_samples))

    for oi, r in enumerate(range(r0_out, r1_out)):
        wt = max(0, r - hw)
        wb = min(nrow - 1, r + hw)
        top_f = max(1, (r + 1) - hw)
        bot_f = min(nrow, (r + 1) + hw)
        for oj, c in enumerate(range(c0_out, c1_out)):
            wl = max(0, c - hw)
            wr = min(ncol - 1, c + hw)
            sub_cov = cover[wt : wb + 1, wl : wr + 1]
            sub_gnd = ground[wt : wb + 1, wl : wr + 1]

            if ibra is not None:
                reg = int(ibra[r, c])
                if reg == 0:
                    out[oi, oj] = -1
                    continue
                mask_reg = ibra[wt : wb + 1, wl : wr + 1] == reg
                n_reg = int(np.sum(mask_reg))
                if n_reg == 0:
                    out[oi, oj] = -1
                    continue
                vc = sub_cov[mask_reg]
                vg = sub_gnd[mask_reg]
                pop_cov = vc[(vc > 0) & (vc <= 101)]
                n_valid = int(np.sum((vc > 0) & (vc <= 101) & (vg > 0) & (vg <= 101)))
                if not _oracle_window_passes(
                    n_reg, len(pop_cov), n_valid, minvalpercent, min_obs_percent, sparse_min_valpercent
                ):
                    out[oi, oj] = -1
                    continue
            else:
                right_f = (c + 1) + hw
                if (c + 1) < hw:
                    npix = (bot_f - top_f) * right_f
                elif right_f > ncol:
                    npix = (bot_f - top_f) * (ncol - (c + 1))
                else:
                    npix = (bot_f - top_f) * window
                vc = sub_cov.ravel()
                vg = sub_gnd.ravel()
                pop_cov = vc[(vc > 0) & (vc <= 101)]
                n_valid = int(np.sum((vc > 0) & (vc <= 101) & (vg > 0) & (vg <= 101)))
                if not _oracle_window_passes(
                    npix, len(pop_cov), n_valid, minvalpercent, min_obs_percent, sparse_min_valpercent
                ):
                    out[oi, oj] = -1
                    continue

            n_pop = len(pop_cov)
            if n_pop == 0:
                out[oi, oj] = -1
                continue
            s_vals = np.sort(pop_cov.astype(np.int64))
            lv = int(lowerquantile * n_pop)
            uv = int(upperquantile * n_pop)
            b0 = b1 = 0
            uvfound = False
            for cand in range(101, 0, -1):
                cnt_le = int(np.searchsorted(s_vals, cand, side="right"))
                if cnt_le < uv and not uvfound:
                    b1 = cand
                    uvfound = True
                if cnt_le < lv:
                    b0 = cand
                    break
            in_band = (vc >= b0 + 1) & (vc <= b1 + 1) & (vg > 0) & (vg <= 101)
            nsum = int(np.sum(in_band))
            if nsum == 0 or nsum < eff_min_samples:
                out[oi, oj] = -1
            else:
                out[oi, oj] = int(np.sum(vg[in_band].astype(np.int64))) // nsum
    return out


def _write_test_geotiff(
    path,
    bands_u8,
    gt,
    proj_wkt,
    nodata=None,
    compress="DEFLATE",
    predictor=1,
    block_size=128,
    interleave="BAND",
):
    """Helper to write a multi-band tiled uint8 GeoTIFF."""
    h, w = bands_u8[0].shape
    drv = gdal.GetDriverByName("GTiff")
    opts = [
        f"COMPRESS={compress}",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
        f"INTERLEAVE={interleave}",
    ]
    if compress in ("DEFLATE", "LZW", "ZSTD"):
        opts.append(f"PREDICTOR={predictor}")
    ds = drv.Create(path, w, h, len(bands_u8), gdal.GDT_Byte, options=opts)
    ds.SetGeoTransform(gt)
    if proj_wkt:
        ds.SetProjection(proj_wkt)
    for idx, arr in enumerate(bands_u8, start=1):
        b = ds.GetRasterBand(idx)
        if nodata is not None:
            b.SetNoDataValue(nodata)
        b.WriteArray(arr)
    ds.FlushCache()
    ds = None


def test_1_utility_functions():
    print("\n--- Test 1: Utility Functions (format_time, derive_output_name) ---")
    assert format_time(0) == "0s"
    assert format_time(42) == "42s"
    assert format_time(59) == "59s"
    assert format_time(60) == "1m 00s"
    assert format_time(125) == "2m 05s"
    assert format_time(3599) == "59m 59s"
    assert format_time(3600) == "1h 00m 00s"
    assert format_time(3665) == "1h 01m 05s"

    assert (
        derive_output_name("/data/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif")
        == "lztmre_aus_m202606202608_doda2_ibra_vinterim.tif"
    )
    assert (
        derive_output_name("/data/GC/lztmre_qld_m201609201611_dixa2_vinterim.tif")
        == "lztmre_qld_m201609201611_doda2_ibra_vinterim.tif"
    )
    assert (
        derive_output_name("/data/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif", no_ibra=True)
        == "lztmre_aus_m202606202608_doda2_vinterim.tif"
    )
    assert (
        derive_output_name("/data/GC/lztmre_qld_m201609201611_dixa2_vinterim.tif", no_ibra=True)
        == "lztmre_qld_m201609201611_doda2_vinterim.tif"
    )
    assert derive_output_name("custom_mosaic.tif") == "custom_mosaic_drcm_ibra.tif"
    assert derive_output_name("custom_mosaic.tif", no_ibra=True) == "custom_mosaic_drcm.tif"
    print(">>> PASS: Utility helpers verified!")


def test_2_groundcover_conversion():
    print("\n--- Test 2: Groundcover Encoding Conversion (LUT, in-place, fallback, empty) ---")
    dea_bare = np.array([[0, 25, 50], [75, 100, 255]], dtype=np.uint8)
    expected_dea = np.array([[101, 76, 51], [26, 1, 0]], dtype=np.int8)

    dea_gc = convert_to_groundcover(dea_bare, nodata_val=255)
    assert np.array_equal(dea_gc, expected_dea)

    dea_copy = dea_bare.copy()
    dea_inplace = convert_to_groundcover(dea_copy, nodata_val=255, inplace=True)
    assert dea_inplace.dtype == np.int8 and np.array_equal(dea_inplace, expected_dea)
    assert np.shares_memory(dea_copy, dea_inplace)

    dea_u16 = convert_to_groundcover(dea_bare.astype(np.uint16), nodata_val=255)
    assert np.array_equal(dea_u16, expected_dea)

    jrsrp_bare = np.array([[100, 125, 150], [175, 200, 0]], dtype=np.uint8)
    expected_jrsrp = np.array([[101, 76, 51], [26, 1, 0]], dtype=np.int8)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare, nodata_val=0), expected_jrsrp)
    jrsrp_copy = jrsrp_bare.copy()
    assert np.array_equal(convert_to_groundcover(jrsrp_copy, nodata_val=0, inplace=True), expected_jrsrp)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare.astype(np.uint16), nodata_val=0), expected_jrsrp)

    assert np.all(convert_to_groundcover(np.full((10, 10), 255, dtype=np.uint8), nodata_val=255) == 0)
    empty_gc = convert_to_groundcover(np.empty((0, 0), dtype=np.uint8), nodata_val=255, inplace=True)
    assert empty_gc.shape == (0, 0) and empty_gc.dtype == np.int8

    # dixa2 containing NoData=255 (uint8 inplace & fallback) + int16 negatives + filename stage
    dixa_with_255 = np.array([[100, 150, 200], [120, 180, 255]], dtype=np.uint8)
    expected_dixa_255 = np.array([[101, 51, 1], [81, 21, 0]], dtype=np.int8)
    assert np.array_equal(convert_to_groundcover(dixa_with_255.copy(), nodata_val=255, inplace=True), expected_dixa_255)
    assert np.array_equal(convert_to_groundcover(dixa_with_255, nodata_val=None), expected_dixa_255)
    assert np.array_equal(convert_to_groundcover(dixa_with_255.astype(np.uint16), nodata_val=255), expected_dixa_255)
    neg_i16 = np.array([[-9999, 0, 50, 100]], dtype=np.int16)
    assert np.array_equal(convert_to_groundcover(neg_i16, nodata_val=255), np.array([[0, 101, 51, 1]], dtype=np.int8))
    assert infer_bare_encoding_from_filename("/tmp/dp8_dir/lztmre_aus_m202001_dixa2.tif") == "dixa2"
    assert infer_bare_encoding_from_filename("/data/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif") == "dp8a2"
    assert infer_bare_encoding_from_filename("/tmp/custom_input.tif") is None
    print(">>> PASS: Groundcover LUT, in-place, filename stage, and fallback verified!")


def test_3_quantile_bounds_and_mean():
    print("\n--- Test 3: Kernel Micro-Functions (_bounds, _bounds_fast, _calcmean) ---")
    hist = np.zeros(104, dtype=np.int32)
    hist[50] = 10
    hist[80] = 10

    b0_hi, b1_hi = refpix_numba_ibra._bounds(hist[:103], 101, 0.90, 0.95)
    b0_hi_f, b1_hi_f = refpix_numba._bounds_fast(hist, 20, 101, 0.90, 0.95)
    assert (b0_hi, b1_hi) == (79, 79) and (b0_hi_f, b1_hi_f) == (79, 79)

    b0_mid, b1_mid = refpix_numba_ibra._bounds(hist[:103], 101, 0.40, 0.60)
    b0_mid_f, b1_mid_f = refpix_numba._bounds_fast(hist, 20, 101, 0.40, 0.60)
    assert (b0_mid, b1_mid) == (49, 79) and (b0_mid_f, b1_mid_f) == (49, 79)

    # Multi-bin weighted average inside [lowerb+1 .. upperb+1] + integer floor rounding
    dryhist = np.zeros(103, dtype=np.int32)
    drysum = np.zeros(103, dtype=np.int32)
    dryhist[50] = 999  # Excluded (below lowerb+1 = 51)
    drysum[50] = 99900
    dryhist[51] = 4
    drysum[51] = 200
    dryhist[52] = 6
    drysum[52] = 407  # Total in [51, 52] = 607 // 10 = 60
    dryhist[53] = 999  # Excluded (above upperb+1 = 52)
    drysum[53] = 99900
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 50, 51) == 60
    assert refpix_numba._calcmean(drysum, dryhist, 50, 51) == 60
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 50, 51, 10) == 60
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 50, 51, 11) == -1
    assert refpix_numba._calcmean(drysum, dryhist, 50, 51, 11) == -1

    dryhist.fill(0)
    drysum.fill(0)
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 10, 20) == -1
    assert refpix_numba._calcmean(drysum, dryhist, 10, 20) == -1
    assert refpix_numba_ibra._bounds(np.zeros(103, dtype=np.int32), 101, 0.90, 0.95) == (0, 0)
    assert refpix_numba._bounds_fast(np.zeros(104, dtype=np.int32), 0, 101, 0.90, 0.95) == (0, 0)
    print(">>> PASS: Quantile bounds, multi-bin mean, and edge cases verified!")


def test_4_oracle_invariance_and_slicing():
    print("\n--- Test 4: Independent Oracle Parity, Single-Region Invariance & Tiled Kernels ---")
    rng = np.random.default_rng(2026)
    oh, ow = 47, 61
    cov_o = rng.integers(0, 102, size=(oh, ow), dtype=np.int8)
    gnd_o = rng.integers(0, 102, size=(oh, ow), dtype=np.int8)
    cov_o[5:12, 10:20] = 0
    gnd_o[20:28, 30:40] = 0
    ibra_o = rng.choice([0, 17, 58, 89], size=(oh, ow)).astype(np.uint8)

    for win in (11, 15):
        oracle_ib = _oracle_refpixel(cov_o, gnd_o, ibra_o, window=win, minvalpercent=0.15)
        numba_ib = refpix_numba_ibra.refpixelfilter_ibra(cov_o, gnd_o, ibra_o, window=win, minvalpercent=0.15)
        assert np.array_equal(oracle_ib, numba_ib), f"IBRA kernel vs independent oracle mismatch (window={win})"

        oracle_gl = _oracle_refpixel(cov_o, gnd_o, None, window=win, minvalpercent=0.15)
        numba_gl = refpix_numba.refpixelfilternodata(cov_o, gnd_o, window=win, minvalpercent=0.15)
        assert np.array_equal(oracle_gl, numba_gl), f"Global kernel vs independent oracle mismatch (window={win})"

    # Large non-square multi-tile & U16/I32 invariance
    np.random.seed(42)
    nrow, ncol = 190, 215
    cover = np.random.randint(0, 102, size=(nrow, ncol), dtype=np.int8)
    ground = np.random.randint(0, 102, size=(nrow, ncol), dtype=np.int8)
    cover[:30, :30] = 0
    ground[:30, :30] = 0
    ibra = np.full((nrow, ncol), 17, dtype=np.uint8)

    window, lq, uq, minval = 31, 0.90, 0.95, 0.15
    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)
    assert int(np.sum(res_global > 0)) > 10000 and int(np.sum(res_global == -1)) > 100
    assert np.array_equal(res_global, res_ibra), "res_global and res_ibra must be bit-for-bit identical"

    r0, r1, c0, c1 = 23, 169, 19, 191
    sliced_small_tiles = refpix_numba.refpixelfilternodata(
        cover, ground, window, lq, uq, minval,
        r_start=r0, r_stop=r1, c_start=c0, c_stop=c1,
        strip_h=32, col_tile_w=48,
    )
    assert np.array_equal(sliced_small_tiles, res_global[r0:r1, c0:c1]), "Sub-window or tile-size mismatch!"

    res_i32 = refpix_numba._KERNEL_I32(
        cover, ground, window, lq, uq, minval, 0, nrow, 0, ncol, 64, 64
    )
    assert np.array_equal(res_i32, res_global), "_KERNEL_I32 vs _KERNEL_U16 mismatch!"
    print(">>> PASS: Independent brute-force oracle, sub-window slicing, tiles, and U16/I32 match 100%!")


def test_5_boundary_isolation():
    print("\n--- Test 5: Row & Column Boundary Isolation Test ---")
    size = 100
    cover = np.zeros((size, size), dtype=np.int8)
    ground = np.zeros((size, size), dtype=np.int8)
    ibra = np.zeros((size, size), dtype=np.uint8)

    # Quadrant-specific IBRA regions + gradients so axis swaps cannot hide
    cover[:50, :50] = 80
    ground[:50, :50] = 70
    ibra[:50, :50] = 10
    cover[:50:6, :50] = 95
    ground[:50:6, :50] = 90

    cover[:50, 50:] = 20
    ground[:50, 50:] = 15
    ibra[:50, 50:] = 20
    cover[:50:6, 50:] = 35
    ground[:50:6, 50:] = 30

    cover[50:, :] = 55
    ground[50:, :] = 45
    ibra[50:, :] = 30
    cover[50::6, :] = 65
    ground[50::6, :] = 60

    window, lq, uq, minval = 21, 0.90, 0.95, 0.05
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)
    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)

    # Vertical split check (IBRA 10 vs 20 across col 49/50)
    assert np.all(res_ibra[10:40, 48:50] == 90)
    assert np.all(res_ibra[10:40, 50:52] == 30)
    assert np.all(res_global[10:40, 50:52] > 60)

    # Horizontal split check (IBRA 20 vs 30 across row 49/50)
    assert np.all(res_ibra[48:50, 60:90] == 30)
    assert np.all(res_ibra[50:52, 60:90] == 60)
    print(">>> PASS: Zero horizontal or vertical cross-boundary contamination proven!")


def test_6_ocean_and_valid_fraction():
    print("\n--- Test 6: Ocean Masking, Empty-Block Skipping & minvalpercent Thresholds ---")
    size = 50
    cover = np.full((size, size), 60, dtype=np.int8)
    ground = np.full((size, size), 50, dtype=np.int8)
    ibra = np.full((size, size), 15, dtype=np.uint8)

    ibra[:25, :25] = 0
    res = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11)
    assert np.all(res[:25, :25] == -1)
    assert np.all(res[35:, 35:] == 50)

    cover.fill(0)
    ground.fill(0)
    ibra.fill(15)
    cover[20:25, 20:26] = 60  # 30 / 121 = 24.8% valid
    ground[20:25, 20:26] = 50

    res_insuff = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11, minvalpercent=0.30)
    res_suff = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11, minvalpercent=0.15)
    assert res_insuff[22, 22] == -1 and res_suff[22, 22] == 50
    assert res_insuff[5, 5] == -1 and res_suff[5, 5] == -1

    sparse_h, sparse_w = 600, 600
    sp_cov = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_gnd = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_cov[300:360, 300:360] = 72
    sp_gnd[300:360, 300:360] = 64

    blocks = refpix_numba._build_valid_blocks(sp_cov, sp_gnd, 256)
    assert blocks[0, 0] == 0 and blocks[1, 1] == 1

    sp_fast = refpix_numba.refpixelfilternodata(sp_cov, sp_gnd, window=31, minvalpercent=0.15)
    sp_cov_unskipped = sp_cov.copy()
    sp_cov_unskipped[0, 0] = 50
    sp_gnd_unskipped = sp_gnd.copy()
    sp_gnd_unskipped[0, 0] = 50
    sp_unskipped = refpix_numba.refpixelfilternodata(
        sp_cov_unskipped, sp_gnd_unskipped, window=31, minvalpercent=0.15
    )
    assert np.all(sp_fast[:250, :250] == -1)
    assert np.all(sp_fast[:, -1] == -1)
    assert np.array_equal(sp_fast[32:, :], sp_unskipped[32:, :])
    assert np.array_equal(sp_fast[:, 32:], sp_unskipped[:, 32:])
    assert np.all(sp_fast[300:360, 300:360] == 64)

    oracle_edge = _oracle_refpixel(
        sp_cov, sp_gnd, None, window=31, minvalpercent=0.15, r_slice=(280, 320), c_slice=(290, 340)
    )
    assert np.array_equal(sp_fast[280:320, 290:340], oracle_edge)
    print(">>> PASS: Ocean masking, empty-block skip, and validity thresholds verified!")


def test_7_local_satellite_multi_ibra():
    print("\n--- Test 7: Portable Real Satellite Multi-IBRA Test (test_data/ chips, offset 40,40) ---")
    assert os.path.exists(LOCAL_GC_CHIP) and os.path.exists(LOCAL_REF_CHIP)
    with tempfile.TemporaryDirectory(prefix="drcm_test7_") as tmpdir:
        out_tif = os.path.join(tmpdir, "local_sat_multi_ibra.tif")
        run_seasonal_drcm_ibra(
            input_groundcover=LOCAL_GC_CHIP,
            percentile_file=LOCAL_REF_CHIP,
            output_file=out_tif,
            window=61,
            chunksize=200,
        )

        ds_res = gdal.Open(out_tif)
        ds_gc = gdal.Open(LOCAL_GC_CHIP)
        ds_ref = gdal.Open(LOCAL_REF_CHIP)
        assert ds_res.RasterCount == 2
        assert ds_res.RasterXSize == ds_gc.RasterXSize == 440
        assert ds_res.RasterYSize == ds_gc.RasterYSize == 440
        assert ds_res.GetGeoTransform() == ds_gc.GetGeoTransform()

        b1, b2 = ds_res.GetRasterBand(1), ds_res.GetRasterBand(2)
        assert "IBRA-Constrained" in b1.GetDescription()
        assert "IBRA-Constrained" in b2.GetDescription()
        assert b1.GetOverviewCount() == 4 and b2.GetOverviewCount() == 4

        dgc = b1.ReadAsArray()
        ref_scaled = b2.ReadAsArray()
        assert int(np.sum(dgc > 0)) > 120000 and int(np.sum(ref_scaled > 0)) > 120000

        # Offset alignment check: GC starts at pixel (40, 40) inside REF chip
        perc = ds_ref.GetRasterBand(1).ReadAsArray(40, 40, 440, 440).astype(np.int8)
        ibra = ds_ref.GetRasterBand(2).ReadAsArray(40, 40, 440, 440)
        bare = ds_gc.GetRasterBand(1).ReadAsArray()
        gc = convert_to_groundcover(bare)

        oracle_sub = _oracle_refpixel(
            perc, gc, ibra, window=61, r_slice=(150, 290), c_slice=(150, 290)
        )
        oracle_ref_sub = np.where(oracle_sub < 0, 0, 100 + oracle_sub).astype(np.uint8)
        oracle_dgc_sub = np.where(
            (oracle_sub < 0) | (gc[150:290, 150:290] == 0), 0, 100 + gc[150:290, 150:290] - oracle_sub
        ).astype(np.uint8)
        assert np.array_equal(ref_scaled[150:290, 150:290], oracle_ref_sub)
        assert np.array_equal(dgc[150:290, 150:290], oracle_dgc_sub)

        # Verify Darling Riverine Plains (17) vs Cobar Peneplain (58) statistical divergence
        assert 17 in np.unique(ibra) and 58 in np.unique(ibra)
        mean_17 = float(np.mean(ref_scaled[(ibra == 17) & (ref_scaled > 0)]))
        mean_58 = float(np.mean(ref_scaled[(ibra == 58) & (ref_scaled > 0)]))
        assert abs(mean_17 - mean_58) > 0.1
        ds_res = ds_gc = ds_ref = None
    print(">>> PASS: Real satellite multi-IBRA chip + spatial offset (40,40) + oracle parity verified!")


def test_8_local_satellite_no_ibra_and_geojson():
    print("\n--- Test 8: Portable Real Satellite --no-ibra & Halo-Padded GeoJSON Test ---")
    assert os.path.exists(LOCAL_AOI_GEOJSON)
    with tempfile.TemporaryDirectory(prefix="drcm_test8_") as tmpdir:
        full_no_ibra = os.path.join(tmpdir, "local_sat_no_ibra.tif")
        full_ibra = os.path.join(tmpdir, "local_sat_ibra.tif")
        aoi_no_ibra = os.path.join(tmpdir, "aoi_no_ibra.tif")
        aoi_ibra = os.path.join(tmpdir, "aoi_ibra.tif")

        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=full_no_ibra,
            window=61, chunksize=200, no_ibra=True, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=full_ibra,
            window=61, chunksize=200, no_ibra=False, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=aoi_no_ibra,
            window=61, chunksize=140, geojson=LOCAL_AOI_GEOJSON, no_ibra=True, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=aoi_ibra,
            window=61, chunksize=140, geojson=LOCAL_AOI_GEOJSON, no_ibra=False, no_overviews=True,
        )

        ds_full_ni = gdal.Open(full_no_ibra)
        assert "Unconstrained" in ds_full_ni.GetRasterBand(1).GetDescription()
        dgc_ni = ds_full_ni.GetRasterBand(1).ReadAsArray()
        ref_ni = ds_full_ni.GetRasterBand(2).ReadAsArray()

        ds_gc = gdal.Open(LOCAL_GC_CHIP)
        ds_ref = gdal.Open(LOCAL_REF_CHIP)
        gt = ds_gc.GetGeoTransform()
        perc = ds_ref.GetRasterBand(1).ReadAsArray(40, 40, 440, 440).astype(np.int8)
        gc = convert_to_groundcover(ds_gc.GetRasterBand(1).ReadAsArray())

        oracle_sub = _oracle_refpixel(perc, gc, None, window=61, r_slice=(160, 280), c_slice=(160, 280))
        exp_ref = np.where(oracle_sub < 0, 0, 100 + oracle_sub).astype(np.uint8)
        exp_dgc = np.where(
            (oracle_sub < 0) | (gc[160:280, 160:280] == 0), 0, 100 + gc[160:280, 160:280] - oracle_sub
        ).astype(np.uint8)
        assert np.array_equal(ref_ni[160:280, 160:280], exp_ref)
        assert np.array_equal(dgc_ni[160:280, 160:280], exp_dgc)

        ds_full_ib = gdal.Open(full_ibra)
        for full_ds, aoi_path in ((ds_full_ni, aoi_no_ibra), (ds_full_ib, aoi_ibra)):
            sub_ds = gdal.Open(aoi_path)
            sgt = sub_ds.GetGeoTransform()
            sw, sh = sub_ds.RasterXSize, sub_ds.RasterYSize
            assert 275 <= sw <= 290 and 275 <= sh <= 290, f"Unexpected AOI subset size: {sw}x{sh}"
            x0 = int(round((sgt[0] - gt[0]) / gt[1]))
            y0 = int(round((sgt[3] - gt[3]) / gt[5]))
            assert 75 <= x0 <= 85 and 75 <= y0 <= 85
            assert np.array_equal(sub_ds.GetRasterBand(1).ReadAsArray(), full_ds.GetRasterBand(1).ReadAsArray()[y0 : y0 + sh, x0 : x0 + sw])
            assert np.array_equal(sub_ds.GetRasterBand(2).ReadAsArray(), full_ds.GetRasterBand(2).ReadAsArray()[y0 : y0 + sh, x0 : x0 + sw])
            sub_ds = None
        ds_full_ni = ds_full_ib = ds_gc = ds_ref = None
    print(">>> PASS: Real satellite --no-ibra + GeoJSON halo AOI verified!")


def test_9_geotiff_formats_prefetch_and_cli():
    print("\n--- Test 9: GeoTIFF Formats (PREDICTOR=1/2, INTERLEAVE, LZW), Prefetch & CLI ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test9_") as tmpdir:
        in_pred1 = os.path.join(tmpdir, "in_pred1.tif")
        in_pred2 = os.path.join(tmpdir, "in_pred2.tif")
        ref_band = os.path.join(tmpdir, "ref_band.tif")
        ref_pixel_lzw = os.path.join(tmpdir, "ref_pixel_lzw.tif")

        rng = np.random.default_rng(12345)
        h, w = 300, 300
        bare = rng.integers(0, 101, size=(h, w), dtype=np.uint8)
        bare[20:40, 20:40] = 255
        ref_arr = rng.integers(1, 102, size=(h, w), dtype=np.uint8)
        ref_arr[128:256, 128:256] = 0
        ibra_arr = rng.choice([17, 58], size=(h, w)).astype(np.uint8)
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)

        _write_test_geotiff(in_pred1, [bare], gt, "", nodata=255, compress="DEFLATE", predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(in_pred2, [bare], gt, "", nodata=255, compress="DEFLATE", predictor=2, block_size=128, interleave="BAND")
        _write_test_geotiff(ref_band, [ref_arr, ibra_arr], gt, "", nodata=0, compress="DEFLATE", predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(ref_pixel_lzw, [ref_arr, ibra_arr], gt, "", nodata=0, compress="LZW", predictor=2, block_size=64, interleave="PIXEL")

        # 1. Verify _LocalBandReader across format combinations & need_ibra=True/False
        for in_p, ref_p in ((in_pred1, ref_band), (in_pred2, ref_pixel_lzw)):
            rdr = _LocalBandReader(in_p, ref_p, ibra_band_idx=2)
            try:
                all_in, all_ref, all_ib = rdr.read_all_windows((0, 0, w, h), (0, 0, w, h), need_ibra=True)
                assert np.array_equal(all_in, bare)
                assert np.array_equal(all_ref, ref_arr.view(np.int8))
                assert np.array_equal(all_ib, ibra_arr)
                in2, ref2, ib2 = rdr.read_all_windows((0, 0, w, h), (0, 0, w, h), need_ibra=False)
                assert np.array_equal(in2, bare) and np.array_equal(ref2, ref_arr.view(np.int8)) and ib2 is None
            finally:
                rdr.close()

        # 2. Multi-chunk end-to-end parity (prefetch + overlap reuse) across formats
        gc = convert_to_groundcover(bare, 255)
        for no_ibra in (True, False):
            if no_ibra:
                direct_ref = refpix_numba.refpixelfilternodata(ref_arr.view(np.int8), gc, window=21)
            else:
                direct_ref = refpix_numba_ibra.refpixelfilter_ibra(ref_arr.view(np.int8), gc, ibra_arr, window=21)
            exp_ref = np.where(direct_ref < 0, 0, 100 + direct_ref).astype(np.uint8)
            exp_dgc = np.where((direct_ref < 0) | (gc == 0), 0, 100 + gc - direct_ref).astype(np.uint8)

            out_p = os.path.join(tmpdir, f"out_fmt_{no_ibra}.tif")
            run_seasonal_drcm_ibra(
                input_groundcover=in_pred2 if no_ibra else in_pred1,
                percentile_file=ref_pixel_lzw if no_ibra else ref_band,
                output_file=out_p,
                window=21,
                chunksize=95,
                no_ibra=no_ibra,
                no_overviews=True,
            )
            ds_out = gdal.Open(out_p)
            assert ds_out.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE") == "2"
            assert np.array_equal(ds_out.GetRasterBand(1).ReadAsArray(), exp_dgc)
            assert np.array_equal(ds_out.GetRasterBand(2).ReadAsArray(), exp_ref)
            ds_out = None

        # 3. CLI entry point & automatic output filename derivation into target directory
        cli_dir = os.path.join(tmpdir, "cli_out") + os.sep
        seasonalDRCMIbra.main([
            LOCAL_GC_CHIP,
            "--percentile-file", LOCAL_REF_CHIP,
            "-o", cli_dir,
            "--window", "21",
            "--chunksize", "220",
            "--min-obs-percent", "0.5",
            "--sparse-min-valpercent", "0.02",
            "--min-ref-samples", "5",
            "--no-overviews",
        ])
        expected_cli_out = os.path.join(tmpdir, "cli_out", "lztmre_aus_m202606202608_doda2_ibra_vinterim.tif")
        assert os.path.isfile(expected_cli_out), f"Expected derived CLI output at {expected_cli_out}"
        ds_cli = gdal.Open(expected_cli_out)
        assert ds_cli.RasterXSize == 440 and ds_cli.RasterYSize == 440 and ds_cli.RasterCount == 2
        ds_cli = None
    print(">>> PASS: GeoTIFF formats, chunk prefetch/overlap & CLI execution verified!")


def test_10_multichunk_and_halo_aoi_parity():
    print("\n--- Test 10: Multi-Chunk Overlap, Spatial Offset Alignment & Strict Halo AOI Parity ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test10_") as tmpdir:
        srs_3577 = osr.SpatialReference()
        srs_3577.ImportFromEPSG(3577)
        srs_3577.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_3577 = srs_3577.ExportToWkt()

        srs_4326 = osr.SpatialReference()
        srs_4326.ImportFromEPSG(4326)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Larger reference grid so input_groundcover has non-zero alignment offset (col=40, row=35)
        ref_h, ref_w = 500, 520
        off_x, off_y = 40, 35
        h, w = 420, 440
        ref_gt = (1000000.0, 30.0, 0.0, -3200000.0, 0.0, -30.0)
        in_gt = (ref_gt[0] + off_x * 30.0, 30.0, 0.0, ref_gt[3] - off_y * 30.0, 0.0, -30.0)

        rng = np.random.default_rng(777)
        bare = rng.integers(5, 95, size=(h, w), dtype=np.uint8)
        bare[:40, :40] = 255
        perc = rng.integers(10, 101, size=(ref_h, ref_w), dtype=np.uint8)
        ibra = np.where(np.arange(ref_w)[None, :] < (off_x + 220), 17, 58).astype(np.uint8)

        in_tif = os.path.join(tmpdir, "aus_synth_dp8a2.tif")
        ref_tif = os.path.join(tmpdir, "aus_synth_ref_ibra.tif")
        _write_test_geotiff(in_tif, [bare], in_gt, wkt_3577, nodata=255, predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(ref_tif, [perc, ibra], ref_gt, wkt_3577, nodata=0, predictor=1, block_size=128, interleave="BAND")

        for no_ibra in (True, False):
            tag = "noibra" if no_ibra else "ibra"
            full_1 = os.path.join(tmpdir, f"full_single_{tag}.tif")
            full_n = os.path.join(tmpdir, f"full_multi_{tag}.tif")
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=full_1, window=31, chunksize=1024,
                no_ibra=no_ibra, no_overviews=True,
            )
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=full_n, window=31, chunksize=96,
                no_ibra=no_ibra, no_overviews=True,
            )
            d1 = gdal.Open(full_1)
            dn = gdal.Open(full_n)
            for b in (1, 2):
                assert np.array_equal(d1.GetRasterBand(b).ReadAsArray(), dn.GetRasterBand(b).ReadAsArray())
            d1 = dn = None

        # AOI cols 150..310 (160 px wide), rows 120..300 (180 px tall) crossing IBRA col 220
        minx = in_gt[0] + 150 * 30.0
        maxx = in_gt[0] + 310 * 30.0
        maxy = in_gt[3] - 120 * 30.0
        miny = in_gt[3] - 300 * 30.0

        # Prove halo padding IS genuinely required (unpadded test_chip misses border pixels)
        nopad_tif = os.path.join(tmpdir, "nopad_chip.tif")
        run_seasonal_drcm_ibra(
            in_tif, ref_tif, output_file=nopad_tif, window=31, chunksize=200,
            test_chip=[150, 120, 160, 180], no_ibra=True, no_overviews=True,
        )
        np_ds = gdal.Open(nopad_tif)
        full_ds_tmp = gdal.Open(os.path.join(tmpdir, "full_single_noibra.tif"))
        assert not np.array_equal(
            np_ds.GetRasterBand(1).ReadAsArray()[0, :],
            full_ds_tmp.GetRasterBand(1).ReadAsArray()[120, 150:310],
        ), "Unpadded crop unexpectedly matched haloed interior!"
        np_ds = full_ds_tmp = None

        import json
        to_4326 = osr.CoordinateTransformation(srs_3577, srs_4326)
        ring_coords = []
        for px, py in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)):
            lon, lat, _ = to_4326.TransformPoint(px, py)
            ring_coords.append([lon, lat])
        geojson_path = os.path.join(tmpdir, "aoi_4326.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    "features": [
                        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring_coords]}}
                    ],
                },
                f,
            )

        for no_ibra in (True, False):
            tag = "noibra" if no_ibra else "ibra"
            ref_full_ds = gdal.Open(os.path.join(tmpdir, f"full_single_{tag}.tif"))
            full_b1 = ref_full_ds.GetRasterBand(1).ReadAsArray()
            full_b2 = ref_full_ds.GetRasterBand(2).ReadAsArray()
            ref_full_ds = None

            out_bbox = os.path.join(tmpdir, f"bbox_{tag}.tif")
            out_gj = os.path.join(tmpdir, f"gj_{tag}.tif")
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=out_bbox, window=31, chunksize=80,
                bbox=[minx, miny, maxx, maxy], no_ibra=no_ibra, no_overviews=True,
            )
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=out_gj, window=31, chunksize=80,
                geojson=geojson_path, no_ibra=no_ibra, no_overviews=True,
            )

            for p in (out_bbox, out_gj):
                sub_ds = gdal.Open(p)
                sgt = sub_ds.GetGeoTransform()
                sw, sh = sub_ds.RasterXSize, sub_ds.RasterYSize
                assert 159 <= sw <= 165 and 179 <= sh <= 185, f"Subset dims {sw}x{sh} out of expected tight AOI range"
                out_minx, out_maxx = sgt[0], sgt[0] + sw * sgt[1]
                out_maxy, out_miny = sgt[3], sgt[3] + sh * sgt[5]
                assert out_minx <= minx + 1e-6 and out_maxx >= maxx - 1e-6
                assert out_maxy >= maxy - 1e-6 and out_miny <= miny + 1e-6
                x0 = int(round((sgt[0] - in_gt[0]) / in_gt[1]))
                y0 = int(round((sgt[3] - in_gt[3]) / in_gt[5]))
                assert np.array_equal(sub_ds.GetRasterBand(1).ReadAsArray(), full_b1[y0 : y0 + sh, x0 : x0 + sw])
                assert np.array_equal(sub_ds.GetRasterBand(2).ReadAsArray(), full_b2[y0 : y0 + sh, x0 : x0 + sw])
                sub_ds = None
    print(">>> PASS: Multi-chunk overlap, offset (40,35) alignment & strict halo AOI parity verified!")


def test_11_serpentine_kernels_and_edge_cases():
    print("\n--- Test 11: Serpentine Kernels (strip_h=16384), 1..255 IBRA & Edge Cases vs Oracle ---")
    rng = np.random.default_rng(20261113)

    # 1. IBRA Serpentine: regions >= 90 (up to 255), k=1 vs k>=2 tight bboxes, cover=105 guard,
    # native r_start/r_stop/c_start/c_stop halo slicing across col_tile_w boundaries, and U16/I32 parity.
    H, W = 95, 135
    cov = rng.integers(1, 102, size=(H, W), dtype=np.int16).astype(np.int8)
    gnd = rng.integers(1, 102, size=(H, W), dtype=np.int16).astype(np.int8)
    cov[rng.random((H, W)) < 0.18] = -1
    gnd[rng.random((H, W)) < 0.12] = -1
    cov[12:18, 22:28] = 105  # out-of-range > 101 must never overflow nbins=104 histogram
    ibra = rng.choice([0, 17, 90, 145, 255], size=(H, W)).astype(np.uint8)
    # Force one pure k=1 tile on left and multi-region k>=2 tile + partial ocean across col_tile_w=64
    ibra[:, :64] = 200
    ibra[:15, :64] = 0
    ibra[-15:, :64] = 0

    for sh, cw in ((16384, 64), (32, 48)):
        got = refpix_numba_ibra.refpixelfilter_ibra(
            cov, gnd, ibra, window=15, r_start=6, r_stop=87, c_start=11, c_stop=122,
            strip_h=sh, col_tile_w=cw,
        )
        exp = _oracle_refpixel(
            cov, gnd, ibra=ibra, window=15, r_slice=(6, 87), c_slice=(11, 122),
        )
        assert np.array_equal(got, exp), f"IBRA Serpentine mismatch for strip_h={sh}, col_tile_w={cw}"

    got_i32 = refpix_numba_ibra._IBRA_KERNEL_I32(
        cov, gnd, ibra, 15, 0.90, 0.95, 0.15, 6, 87, 11, 122, 16384, 64,
    )
    assert np.array_equal(got_i32, exp), "_IBRA_KERNEL_I32 vs oracle mismatch!"

    # 2. --no-ibra Serpentine: cover=105 guard, ocean strip trimming, turnaround rows, and U16/I32 parity
    cov_ni = rng.integers(1, 102, size=(120, 140), dtype=np.int16).astype(np.int8)
    gnd_ni = rng.integers(1, 102, size=(120, 140), dtype=np.int16).astype(np.int8)
    cov_ni[:20, :] = -1
    cov_ni[-20:, :] = -1
    cov_ni[35:45, 50:70] = 105
    gnd_ni[rng.random((120, 140)) < 0.15] = -1

    for sh, cw in ((16384, 64), (24, 48)):
        got_ni = refpix_numba.refpixelfilternodata(
            cov_ni, gnd_ni, window=17, r_start=10, r_stop=110, c_start=12, c_stop=128,
            strip_h=sh, col_tile_w=cw,
        )
        exp_ni = _oracle_refpixel(
            cov_ni, gnd_ni, ibra=None, window=17, r_slice=(10, 110), c_slice=(12, 128),
        )
        assert np.array_equal(got_ni, exp_ni), f"No-IBRA Serpentine mismatch for strip_h={sh}, col_tile_w={cw}"

    # 3. Adversarial edge shapes & parameter guards for both kernels
    edge_configs = [
        ("window_1", (27, 31), 1, 0.15, 0, 27, 0, 31),
        ("nrow_1", (1, 75), 11, 0.15, 0, 1, 4, 69),
        ("ncol_lt_half", (38, 5), 21, 0.15, 3, 35, 0, 5),
        ("minval_0", (31, 33), 9, 0.0, 2, 29, 2, 31),
        ("empty_slice", (30, 30), 11, 0.15, 15, 15, 5, 25),
    ]
    for label, (eh, ew), win, mv, rs, rp, cs, cp in edge_configs:
        ec = rng.integers(1, 102, size=(eh, ew), dtype=np.int16).astype(np.int8)
        eg = rng.integers(1, 102, size=(eh, ew), dtype=np.int16).astype(np.int8)
        ec[rng.random((eh, ew)) < 0.25] = -1
        ei = rng.choice([0, 3, 210], size=(eh, ew)).astype(np.uint8)

        g_no = refpix_numba.refpixelfilternodata(
            ec, eg, window=win, minvalpercent=mv, r_start=rs, r_stop=rp, c_start=cs, c_stop=cp,
        )
        e_no = _oracle_refpixel(ec, eg, None, window=win, minvalpercent=mv, r_slice=(rs, rp), c_slice=(cs, cp))
        assert np.array_equal(g_no, e_no), f"No-IBRA edge case {label} mismatch"

        g_ib = refpix_numba_ibra.refpixelfilter_ibra(
            ec, eg, ei, window=win, minvalpercent=mv, r_start=rs, r_stop=rp, c_start=cs, c_stop=cp,
        )
        e_ib = _oracle_refpixel(ec, eg, ei, window=win, minvalpercent=mv, r_slice=(rs, rp), c_slice=(cs, cp))
        assert np.array_equal(g_ib, e_ib), f"IBRA edge case {label} mismatch"

    # 4. int64 sum overflow & out-of-range ground guard: window=4621 (4621^2*101 > 2^31-1) & ground=120
    cov_big = np.full((4621, 4621), 50, dtype=np.int8)
    gnd_big = np.full((4621, 4621), 101, dtype=np.int8)
    ib_big = np.full((4621, 4621), 17, dtype=np.uint8)
    res_big_no = refpix_numba.refpixelfilternodata(
        cov_big, gnd_big, window=4621, minvalpercent=0.1, r_start=2310, r_stop=2312, c_start=2310, c_stop=2312,
    )
    res_big_ib = refpix_numba_ibra.refpixelfilter_ibra(
        cov_big, gnd_big, ib_big, window=4621, minvalpercent=0.1, r_start=2310, r_stop=2312, c_start=2310, c_stop=2312,
    )
    assert np.all(res_big_no == 101) and np.all(res_big_ib == 101), "int64 sum regression on window=4621"
    del cov_big, gnd_big, ib_big

    cov_g = rng.integers(1, 102, size=(41, 41), dtype=np.int16).astype(np.int8)
    gnd_g = rng.integers(1, 102, size=(41, 41), dtype=np.int16).astype(np.int8)
    ib_g = np.full((41, 41), 17, dtype=np.uint8)
    gnd_g[10:30, 10:30] = 120  # invalid > 101 groundcover must be ignored like <= 0
    assert np.array_equal(
        refpix_numba.refpixelfilternodata(cov_g, gnd_g, window=15, minvalpercent=0.15),
        _oracle_refpixel(cov_g, gnd_g, None, window=15, minvalpercent=0.15),
    )
    assert np.array_equal(
        refpix_numba_ibra.refpixelfilter_ibra(cov_g, gnd_g, ib_g, window=15, minvalpercent=0.15),
        _oracle_refpixel(cov_g, gnd_g, ib_g, window=15, minvalpercent=0.15),
    )

    print(">>> PASS: Serpentine IBRA (1..255, k=1/k>=2), No-IBRA (strip_h=16384), int64/bounds & edge cases match oracle 100%!")


def test_12_stage1_offgrid_and_dixa2_pipelines():
    print("\n--- Test 12: dixa2 Chunk Parity, Off-Grid Zero-Padding & CRS Checks, Stage-1 Masking & Reprojection ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test12_") as tmpdir:
        srs_3577 = osr.SpatialReference()
        srs_3577.ImportFromEPSG(3577)
        srs_3577.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_3577 = srs_3577.ExportToWkt()

        srs_4326 = osr.SpatialReference()
        srs_4326.ImportFromEPSG(4326)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_4326 = srs_4326.ExportToWkt()

        # 1. dixa2 GeoTIFF with NoData=255 in chunk 0 and all-(100/255) tail chunk
        rng = np.random.default_rng(1313)
        h_d, w_d = 160, 128
        dix_bare = rng.integers(100, 201, size=(h_d, w_d), dtype=np.uint8)
        dix_bare[:20, :] = 255
        dix_bare[80:, :] = 100
        dix_bare[100:120, :] = 255
        ref_d = rng.integers(1, 102, size=(h_d, w_d), dtype=np.uint8)
        ibra_d = np.full((h_d, w_d), 17, dtype=np.uint8)
        gt_d = (0.0, 25.0, 0.0, 0.0, 0.0, -25.0)
        dix_in_tif = os.path.join(tmpdir, "synth_m202001_dixa2_vinterim.tif")
        anon_dix_tif = os.path.join(tmpdir, "anon_synth_input.tif")
        ref_d_tif = os.path.join(tmpdir, "synth_dixa2_ref_ibra.tif")
        _write_test_geotiff(dix_in_tif, [dix_bare], gt_d, wkt_3577, nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(anon_dix_tif, [dix_bare], gt_d, wkt_3577, nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(ref_d_tif, [ref_d, ibra_d], gt_d, wkt_3577, nodata=0, predictor=1, block_size=128)

        gc_d_exp = convert_to_groundcover(dix_bare, nodata_val=255, encoding="dixa2")
        oracle_d = _oracle_refpixel(ref_d.view(np.int8), gc_d_exp, ibra_d, window=21, minvalpercent=0.10)
        exp_d_ref = np.where(oracle_d < 0, 0, 100 + oracle_d).astype(np.uint8)
        exp_d_dgc = np.where((oracle_d < 0) | (gc_d_exp == 0), 0, 100 + gc_d_exp - oracle_d).astype(np.uint8)

        for label, in_p in (("dixa2_name", dix_in_tif), ("anon_locked", anon_dix_tif)):
            out_d = os.path.join(tmpdir, f"out_{label}.tif")
            run_seasonal_drcm_ibra(
                in_p, ref_d_tif, output_file=out_d, window=21, chunksize=64,
                minvalpercent=0.10, no_overviews=True,
            )
            ds_o = gdal.Open(out_d)
            assert np.array_equal(ds_o.GetRasterBand(1).ReadAsArray(), exp_d_dgc), f"dixa2 DGC mismatch ({label})"
            assert np.array_equal(ds_o.GetRasterBand(2).ReadAsArray(), exp_d_ref), f"dixa2 Ref mismatch ({label})"
            ds_o = None

        # 2. Input extends outside smaller reference (all 4 borders), separate --ibra-raster,
        # plus pixel-resolution and CRS mismatch validation
        in_h, in_w = 100, 100
        ref_h, ref_w = 60, 60
        ref_ox, ref_oy = 15, 20
        in_gt = (100000.0, 25.0, 0.0, -2000000.0, 0.0, -25.0)
        ref_gt = (in_gt[0] + ref_ox * 25.0, 25.0, 0.0, in_gt[3] - ref_oy * 25.0, 0.0, -25.0)
        ib_ox, ib_oy, ib_h, ib_w = 10, 10, 80, 75
        ib_gt = (in_gt[0] + ib_ox * 25.0, 25.0, 0.0, in_gt[3] - ib_oy * 25.0, 0.0, -25.0)

        in_bare_off = rng.integers(10, 90, size=(in_h, in_w), dtype=np.uint8)
        ref_small = rng.integers(10, 101, size=(ref_h, ref_w), dtype=np.uint8)
        ib_small_combined = rng.choice([17, 58], size=(ref_h, ref_w)).astype(np.uint8)
        ib_sep = rng.choice([17, 58], size=(ib_h, ib_w)).astype(np.uint8)

        in_off_tif = os.path.join(tmpdir, "in_off_dp8a2.tif")
        ref_2b_tif = os.path.join(tmpdir, "ref_2b.tif")
        ref_1b_tif = os.path.join(tmpdir, "ref_1b.tif")
        ib_sep_tif = os.path.join(tmpdir, "ib_sep.tif")
        _write_test_geotiff(in_off_tif, [in_bare_off], in_gt, wkt_3577, nodata=255, predictor=1, block_size=32)
        _write_test_geotiff(ref_2b_tif, [ref_small, ib_small_combined], ref_gt, wkt_3577, nodata=0, predictor=1, block_size=32)
        _write_test_geotiff(ref_1b_tif, [ref_small], ref_gt, wkt_3577, nodata=0, predictor=2, block_size=64)
        _write_test_geotiff(ib_sep_tif, [ib_sep], ib_gt, wkt_3577, nodata=0, predictor=1, block_size=32)

        gc_off = convert_to_groundcover(in_bare_off, nodata_val=255, encoding="dp8a2")
        pad_ref = np.zeros((in_h, in_w), dtype=np.int8)
        pad_ref[ref_oy : ref_oy + ref_h, ref_ox : ref_ox + ref_w] = ref_small.view(np.int8)
        pad_ib_comb = np.zeros((in_h, in_w), dtype=np.uint8)
        pad_ib_comb[ref_oy : ref_oy + ref_h, ref_ox : ref_ox + ref_w] = ib_small_combined
        pad_ib_sep = np.zeros((in_h, in_w), dtype=np.uint8)
        pad_ib_sep[ib_oy : ib_oy + ib_h, ib_ox : ib_ox + ib_w] = ib_sep

        for no_ibra, r_path, ib_path, ib_full in (
            (True, ref_2b_tif, None, None),
            (False, ref_2b_tif, None, pad_ib_comb),
            (False, ref_1b_tif, ib_sep_tif, pad_ib_sep),
        ):
            out_off = os.path.join(tmpdir, f"off_{no_ibra}_{ib_path is not None}.tif")
            run_seasonal_drcm_ibra(
                in_off_tif, r_path, ibra_raster=ib_path, output_file=out_off,
                window=15, chunksize=40, minvalpercent=0.10, no_ibra=no_ibra, no_overviews=True,
            )
            o_off = _oracle_refpixel(pad_ref, gc_off, ib_full, window=15, minvalpercent=0.10)
            e_ref = np.where(o_off < 0, 0, 100 + o_off).astype(np.uint8)
            e_dgc = np.where((o_off < 0) | (gc_off == 0), 0, 100 + gc_off - o_off).astype(np.uint8)
            ds_off = gdal.Open(out_off)
            assert np.array_equal(ds_off.GetRasterBand(1).ReadAsArray(), e_dgc)
            assert np.array_equal(ds_off.GetRasterBand(2).ReadAsArray(), e_ref)
            ds_off = None

        bad_res_tif = os.path.join(tmpdir, "bad_res.tif")
        bad_crs_tif = os.path.join(tmpdir, "bad_crs.tif")
        _write_test_geotiff(bad_res_tif, [ref_small], (ref_gt[0], 30.0, 0.0, ref_gt[3], 0.0, -30.0), wkt_3577)
        _write_test_geotiff(bad_crs_tif, [ref_small], ref_gt, wkt_4326)
        for bad_ref in (bad_res_tif, bad_crs_tif):
            got_err = False
            try:
                run_seasonal_drcm_ibra(in_off_tif, bad_ref, output_file=os.path.join(tmpdir, "never.tif"), no_ibra=True, no_overviews=True)
            except ValueError:
                got_err = True
            assert got_err, f"Expected ValueError for misaligned/mismatched raster {bad_ref}"

        # 3. Stage-1 masking: NumPy-2 underflow guards (202/254/255 -> 0), Band 2 IBRA on bare=0 hole,
        # dp8a2 + custom --ibra-attr, OBJECTID > 255 guard, and ensure_ibra_albers reprojection/cache
        s1_h, s1_w = 64, 64
        s1_gt = (0.0, 25.0, 0.0, 0.0, 0.0, -25.0)
        chg = np.zeros((3, s1_h, s1_w), dtype=np.uint8)
        chg[0, 5:10, 5:10] = 2  # forest excluded
        chg[2, 10:15, 5:10] = 1  # cleared excluded
        trn = np.full((1, s1_h, s1_w), 20, dtype=np.uint8)
        trn[0, 15:20, 5:10] = 40  # slope >= 40 excluded
        bare_s1 = np.zeros((7, s1_h, s1_w), dtype=np.uint8)
        bare_s1[6, :, :] = 150
        bare_s1[6, 0:4, 0:4] = 255   # invalid border must map to 0, NOT 101
        bare_s1[6, 0:4, 4:8] = 254   # invalid > 200 must map to 0, NOT 101
        bare_s1[6, 0:4, 8:12] = 202  # invalid > 200 must map to 0, NOT 101
        bare_s1[6, 20:40, 20:40] = 0 # bareStats == 0 land hole inside polygon

        chg_tif = os.path.join(tmpdir, "chg.tif")
        trn_tif = os.path.join(tmpdir, "trn.tif")
        bare_tif = os.path.join(tmpdir, "bare_dixa2.tif")
        _write_test_geotiff(chg_tif, [chg[0], chg[1], chg[2]], s1_gt, wkt_3577)
        _write_test_geotiff(trn_tif, [trn[0]], s1_gt, wkt_3577)
        _write_test_geotiff(bare_tif, [bare_s1[i] for i in range(7)], s1_gt, wkt_3577, nodata=0)

        fgb_ok = os.path.join(tmpdir, "ibra_ok.fgb")
        fgb_bad = os.path.join(tmpdir, "ibra_bad.fgb")
        for fgb_path, oid_val in ((fgb_ok, 17), (fgb_bad, 300)):
            vdrv = ogr.GetDriverByName("FlatGeobuf")
            vds = vdrv.CreateDataSource(fgb_path)
            vlyr = vds.CreateLayer("ibra", srs=srs_3577, geom_type=ogr.wkbPolygon)
            vlyr.CreateField(ogr.FieldDefn("OBJECTID", ogr.OFTInteger))
            feat = ogr.Feature(vlyr.GetLayerDefn())
            feat.SetField("OBJECTID", oid_val)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for px, py in ((0.0, 0.0), (1600.0, 0.0), (1600.0, -1600.0), (0.0, -1600.0), (0.0, 0.0)):
                ring.AddPoint_2D(px, py)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feat.SetGeometry(poly)
            vlyr.CreateFeature(feat)
            vds = None

        s1_out = os.path.join(tmpdir, "s1_out.tif")
        orig_argv_s1 = sys.argv
        try:
            sys.argv = [
                "maskChangeSteepIbra.py",
                "--change", chg_tif,
                "--terrain", trn_tif,
                "--bare", bare_tif,
                "--ibra", fgb_ok,
                "-o", s1_out,
            ]
            maskChangeSteepIbra.main()
        finally:
            sys.argv = orig_argv_s1

        ds_s1 = gdal.Open(s1_out)
        s1_b1 = ds_s1.GetRasterBand(1).ReadAsArray()
        s1_b2 = ds_s1.GetRasterBand(2).ReadAsArray()
        ds_s1 = None
        assert np.all(s1_b1[0:4, 0:12] == 0), f"Stage-1 wraparound bug on 202/254/255: {np.unique(s1_b1[0:4, 0:12])}"
        assert np.all(s1_b1[5:10, 5:10] == 0) and np.all(s1_b1[10:15, 5:10] == 0) and np.all(s1_b1[15:20, 5:10] == 0)
        assert np.all(s1_b1[25:35, 45:55] == 51)
        assert np.all(s1_b2[20:40, 20:40] == 17), "Stage-1 Band 2 IBRA must persist on bareStats==0 land holes"

        bad_ds = ogr.Open(fgb_bad)
        raised_oid = False
        try:
            maskChangeSteepIbra.validate_ibra_layer_attribute(bad_ds.GetLayer(0), "OBJECTID")
        except ValueError:
            raised_oid = True
        bad_ds = None
        assert raised_oid, "Expected ValueError when OBJECTID > 255"

        # Verify ensure_ibra_albers reprojection from EPSG:4326 -> EPSG:3577 + cache hit
        fgb_4326 = os.path.join(tmpdir, "ibra_4326.fgb")
        fgb_albers = os.path.join(tmpdir, "ibra_4326_albers.fgb")
        vdrv = ogr.GetDriverByName("FlatGeobuf")
        vds = vdrv.CreateDataSource(fgb_4326)
        vlyr = vds.CreateLayer("ibra", srs=srs_4326, geom_type=ogr.wkbPolygon)
        vlyr.CreateField(ogr.FieldDefn("OBJECTID", ogr.OFTInteger))
        feat = ogr.Feature(vlyr.GetLayerDefn())
        feat.SetField("OBJECTID", 58)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for lon, lat in ((145.0, -31.0), (146.0, -31.0), (146.0, -32.0), (145.0, -32.0), (145.0, -31.0)):
            ring.AddPoint_2D(lon, lat)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat.SetGeometry(poly)
        vlyr.CreateFeature(feat)
        vds = None

        reproj_path = maskChangeSteepIbra.ensure_ibra_albers(fgb_4326, wkt_3577, output_albers_path=fgb_albers)
        assert reproj_path == fgb_albers and os.path.exists(fgb_albers)
        rds = ogr.Open(fgb_albers)
        rsrs = rds.GetLayer(0).GetSpatialRef()
        assert rsrs is not None and rsrs.IsSame(srs_3577), "ensure_ibra_albers output must match target EPSG:3577"
        rds = None
        # Second call hits existing cache cleanly
        assert maskChangeSteepIbra.ensure_ibra_albers(fgb_4326, wkt_3577, output_albers_path=fgb_albers) == fgb_albers

        # Verify dp8a2 + custom --ibra-attr + --test-chip in maskChangeSteepIbra
        bare_dp8 = np.zeros((7, s1_h, s1_w), dtype=np.uint8)
        bare_dp8[6, :, :] = 40  # dp8a2 bare=40 -> 101 - 40 = 61
        bare_dp8_tif = os.path.join(tmpdir, "bare_dp8a2.tif")
        _write_test_geotiff(bare_dp8_tif, [bare_dp8[i] for i in range(7)], s1_gt, wkt_3577, nodata=255)
        fgb_custom = os.path.join(tmpdir, "ibra_custom.fgb")
        vds = vdrv.CreateDataSource(fgb_custom)
        vlyr = vds.CreateLayer("ibra", srs=srs_3577, geom_type=ogr.wkbPolygon)
        vlyr.CreateField(ogr.FieldDefn("REGION_ID", ogr.OFTInteger))
        feat = ogr.Feature(vlyr.GetLayerDefn())
        feat.SetField("REGION_ID", 205)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for px, py in ((0.0, 0.0), (1600.0, 0.0), (1600.0, -1600.0), (0.0, -1600.0), (0.0, 0.0)):
            ring.AddPoint_2D(px, py)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat.SetGeometry(poly)
        vlyr.CreateFeature(feat)
        vds = None

        s1_dp8_out = os.path.join(tmpdir, "s1_dp8_out.tif")
        orig_argv_s1 = sys.argv
        try:
            sys.argv = [
                "maskChangeSteepIbra.py",
                "--change", chg_tif,
                "--terrain", trn_tif,
                "--bare", bare_dp8_tif,
                "--ibra", fgb_custom,
                "--ibra-attr", "REGION_ID",
                "--bare-encoding", "dp8a2",
                "--test-chip", "0", "0", "32", "32",
                "-o", s1_dp8_out,
            ]
            maskChangeSteepIbra.main()
        finally:
            sys.argv = orig_argv_s1

        ds_dp8 = gdal.Open(s1_dp8_out)
        assert ds_dp8.RasterXSize == 32 and ds_dp8.RasterYSize == 32
        assert np.all(ds_dp8.GetRasterBand(1).ReadAsArray()[22:30, 22:30] == 61)
        assert np.all(ds_dp8.GetRasterBand(2).ReadAsArray() == 205)
        ds_dp8 = None
    print(">>> PASS: dixa2/255 chunks, off-grid/CRS validation, Stage-1 NumPy2 & ensure_ibra_albers verified!")


def test_13_sparse_ref_and_tail_safety():
    print("\n--- Test 13: Sparse-Ref Cloud-Free Recovery, Sample Floor & Empty-Tail NoData ---")
    # 1. Sparse cloud-free window (~3.92% static density < 10%) vs cloudy fragments & sample floor
    win = 101
    h, w = 121, 121
    cy, cx = 60, 60
    cov = np.zeros((h, w), dtype=np.int8)
    gnd = np.zeros((h, w), dtype=np.int8)
    ib = np.full((h, w), 9, dtype=np.uint8)

    # Place 400 static ref pixels inside [10:111, 10:111] (3.92% of 10,201):
    # 300 at cover=40 (ground=45), 100 at cover=80 (ground=70 -> inside [P90, P95]).
    cov[15:30, 15:35] = 40
    gnd[15:30, 15:35] = 45
    cov[40:50, 40:50] = 80
    gnd[40:50, 40:50] = 70

    # Legacy min_obs_percent=0.0 rejects 3.92% (< 0.10)
    assert refpix_numba_ibra.refpixelfilter_ibra(
        cov, gnd, ib, window=win, minvalpercent=0.10, min_obs_percent=0.0, min_ref_samples=0
    )[cy, cx] == -1
    assert refpix_numba.refpixelfilternodata(
        cov, gnd, window=win, minvalpercent=0.10, min_obs_percent=0.0, min_ref_samples=0
    )[cy, cx] == -1

    # Sparse cloud-free recovery (100% obs >= 0.50, 3.92% >= 0.02, nsum=100 >= 50) recovers 70
    for res in (
        refpix_numba_ibra.refpixelfilter_ibra(
            cov, gnd, ib, window=win, minvalpercent=0.10,
            min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=50,
        ),
        refpix_numba.refpixelfilternodata(
            cov, gnd, window=win, minvalpercent=0.10,
            min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=50,
        ),
    ):
        assert res[cy, cx] == 70, f"Expected recovered ref=70, got {res[cy, cx]}"

    # Below sparse_min_valpercent (3.92% < 0.05) stays NoData (-1)
    assert refpix_numba_ibra.refpixelfilter_ibra(
        cov, gnd, ib, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.05, min_ref_samples=50,
    )[cy, cx] == -1
    assert refpix_numba.refpixelfilternodata(
        cov, gnd, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.05, min_ref_samples=50,
    )[cy, cx] == -1

    # Tail floor exact boundary: nsum=100 passes at 100 and fails (-1) at 101
    assert refpix_numba_ibra.refpixelfilter_ibra(
        cov, gnd, ib, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=100,
    )[cy, cx] == 70
    assert refpix_numba_ibra.refpixelfilter_ibra(
        cov, gnd, ib, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=101,
    )[cy, cx] == -1
    assert refpix_numba.refpixelfilternodata(
        cov, gnd, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=101,
    )[cy, cx] == -1

    # Cloudy sparse fragment: 25% seasonally clear (< 0.50) rejected even with min_ref_samples=1
    gnd_cloudy = np.zeros_like(gnd)
    gnd_cloudy[15:30, 15:20] = 45
    gnd_cloudy[40:45, 40:45] = 70
    assert refpix_numba_ibra.refpixelfilter_ibra(
        cov, gnd_cloudy, ib, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=1,
    )[cy, cx] == -1
    assert refpix_numba.refpixelfilternodata(
        cov, gnd_cloudy, window=win, minvalpercent=0.10,
        min_obs_percent=0.50, sparse_min_valpercent=0.02, min_ref_samples=1,
    )[cy, cx] == -1

    # 2. Empty [P90, P95] tail (nsum == 0) must return -1 and encode uint8(0) NoData (never uint8(100))
    cov_et = np.zeros((35, 35), dtype=np.int8)
    gnd_et = np.full((35, 35), 60, dtype=np.int8)
    ib_et = np.full((35, 35), 9, dtype=np.uint8)
    cov_et[5:25, 5:25] = 40   # 400 px
    cov_et[25:30, 5:25] = 85  # 100 px -> [P90, P95] falls in 85
    gnd_et[25:30, 5:25] = 0   # Cloud over [P90, P95] -> nsum == 0, while 400 >= 0.15 * 961 passes
    et_ib = refpix_numba_ibra.refpixelfilter_ibra(cov_et, gnd_et, ib_et, window=31, minvalpercent=0.15)
    et_gl = refpix_numba.refpixelfilternodata(cov_et, gnd_et, window=31, minvalpercent=0.15)
    assert et_ib[17, 17] == -1 and et_gl[17, 17] == -1, "Empty [P90, P95] tail must return -1"

    with tempfile.TemporaryDirectory(prefix="drcm_test13_") as tmpdir:
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)
        bare_et = np.where(gnd_et == 0, 255, 101 - gnd_et).astype(np.uint8)
        in_et = os.path.join(tmpdir, "lztmre_aus_m202603202605_dp8a2.tif")
        ref_et = os.path.join(tmpdir, "ref_et.tif")
        out_et_ib = os.path.join(tmpdir, "out_et_ib.tif")
        out_et_gl = os.path.join(tmpdir, "out_et_gl.tif")
        _write_test_geotiff(in_et, [bare_et], gt, "", nodata=255, predictor=1, block_size=32)
        _write_test_geotiff(ref_et, [cov_et.view(np.uint8), ib_et], gt, "", nodata=0, predictor=1, block_size=32)

        seasonalDRCMIbra.main([
            in_et, "--percentile-file", ref_et,
            "-o", out_et_ib, "--window", "31", "--minvalpercent", "0.15",
            "--min-obs-percent", "0.50", "--sparse-min-valpercent", "0.02",
            "--min-ref-samples", "0", "--no-overviews",
        ])
        seasonalDRCMIbra.main([
            in_et, "--percentile-file", ref_et,
            "-o", out_et_gl, "--window", "31", "--minvalpercent", "0.15",
            "--min-obs-percent", "0.50", "--sparse-min-valpercent", "0.02",
            "--min-ref-samples", "0", "--no-ibra", "--no-overviews",
        ])

        for p_out in (out_et_ib, out_et_gl):
            ds = gdal.Open(p_out)
            b1 = ds.GetRasterBand(1).ReadAsArray()
            b2 = ds.GetRasterBand(2).ReadAsArray()
            ds = None
            assert b1[17, 17] == 0 and b2[17, 17] == 0, (
                f"Expected NoData uint8(0) on empty tail pixel, got b1={b1[17, 17]}, b2={b2[17, 17]}"
            )

    # 3. Randomized fuzz parity across both kernels against independent brute-force oracle
    rng = np.random.default_rng(20261115)
    param_combos = [
        (0.10, 0.50, 0.02, 4),
        (0.25, 0.40, 0.05, 8),
        (0.15, 0.80, 0.01, 1),
        (0.00, 0.00, 0.00, 3),
    ]
    for idx, (mv, mo, smv, mrs) in enumerate(param_combos):
        fh, fw = 43 + idx * 3, 51 + idx * 2
        fcov = rng.integers(0, 108, size=(fh, fw), dtype=np.int16).astype(np.int8)
        fgnd = rng.integers(0, 108, size=(fh, fw), dtype=np.int16).astype(np.int8)
        fcov[rng.random((fh, fw)) < 0.65] = 0
        fgnd[rng.random((fh, fw)) < 0.35] = 0
        fib = rng.choice([0, 9, 17, 89, 210], size=(fh, fw)).astype(np.uint8)
        rs, rp, cs, cp = 4, fh - 3, 5, fw - 4
        win_f = 15 + (idx % 2) * 4

        g_no = refpix_numba.refpixelfilternodata(
            fcov, fgnd, window=win_f, minvalpercent=mv,
            r_start=rs, r_stop=rp, c_start=cs, c_stop=cp,
            min_obs_percent=mo, sparse_min_valpercent=smv, min_ref_samples=mrs,
        )
        e_no = _oracle_refpixel(
            fcov, fgnd, None, window=win_f, minvalpercent=mv,
            r_slice=(rs, rp), c_slice=(cs, cp),
            min_obs_percent=mo, sparse_min_valpercent=smv, min_ref_samples=mrs,
        )
        assert np.array_equal(g_no, e_no), f"Sparse fuzz No-IBRA mismatch combo {idx}"

        g_ib = refpix_numba_ibra.refpixelfilter_ibra(
            fcov, fgnd, fib, window=win_f, minvalpercent=mv,
            r_start=rs, r_stop=rp, c_start=cs, c_stop=cp,
            min_obs_percent=mo, sparse_min_valpercent=smv, min_ref_samples=mrs,
        )
        e_ib = _oracle_refpixel(
            fcov, fgnd, fib, window=win_f, minvalpercent=mv,
            r_slice=(rs, rp), c_slice=(cs, cp),
            min_obs_percent=mo, sparse_min_valpercent=smv, min_ref_samples=mrs,
        )
        assert np.array_equal(g_ib, e_ib), f"Sparse fuzz IBRA mismatch combo {idx}"

    print(">>> PASS: Sparse-ref cloud-free recovery, sample floor, empty-tail NoData & fuzz parity verified!")


def main():
    t0 = time.perf_counter()
    test_1_utility_functions()
    test_2_groundcover_conversion()
    test_3_quantile_bounds_and_mean()
    test_4_oracle_invariance_and_slicing()
    test_5_boundary_isolation()
    test_6_ocean_and_valid_fraction()
    test_7_local_satellite_multi_ibra()
    test_8_local_satellite_no_ibra_and_geojson()
    test_9_geotiff_formats_prefetch_and_cli()
    test_10_multichunk_and_halo_aoi_parity()
    test_11_serpentine_kernels_and_edge_cases()
    test_12_stage1_offgrid_and_dixa2_pipelines()
    test_13_sparse_ref_and_tail_safety()

    t_total = time.perf_counter() - t0
    print("\n========================================================")
    print(f"ALL 13 TESTS (incl. test_data/ real satellite chips) PASSED in {t_total:.2f}s!")
    print("========================================================")


if __name__ == "__main__":
    main()
