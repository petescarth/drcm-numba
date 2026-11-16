# Dynamic Reference Cover Method (DRCM) Pipeline

This repository provides a high-performance, pure-Python + Numba implementation of the Australian seasonal Groundcover Reference Pixel Filter and Dynamic Reference-Cover Method (DRCM) pipeline, based on **Bastin et al. (2012)** ([DOI: 10.1016/j.rse.2012.02.021](https://doi.org/10.1016/j.rse.2012.02.021)).

It modernizes the original Joint Remote Sensing Research Program (JRSRP) codebase ([`jrsrp/themes/cover/groundcover/drcm`](https://gitlab.com/jrsrp/themes/cover/groundcover/drcm))—replacing the compiled Fortran 90 module (`refpixelfilter.f90`) and legacy scripts (`seasonalDCRM.py`, `maskChangeSteep.py`)—with Serpentine 2D tiled Numba JIT kernels, IBRA biogeographic stratification (`1..255` region IDs), halo-padded AOI subsetting, sparse-reference cloud recovery, and a self-contained offline verification suite with real satellite test chips (`test_data/`).

---

## 1. Scientific Foundation: The Dynamic Reference-Cover Method

The Dynamic Reference-Cover Method (DRCM) was established by **Bastin et al. (2012)** to separate grazing management effects from climate variability across rangelands at regional to continental scales.

### A. The Rangeland Monitoring Challenge
- **Climate vs. Management Confounding**: In arid and semi-arid rangelands, high inter-annual rainfall variability ("boom and bust" cycles) masks or confounds management-driven changes in ground cover (grazing pressure, stocking rates, spelling).
- **Scarcity of Physical Benchmarks**: Establishing static, ungrazed reference exclosures across millions of square kilometers is logistically and economically unfeasible, and fixed exclosures cannot capture local environmental gradients everywhere.
- **Absence of Management Spatial Data**: Regional monitoring cannot rely consistently on property-level management GIS records at landscape scale.

### B. Conceptual Framework and Methodology
DRCM establishes a dynamic, image-derived reference framework using multitemporal satellite fractional ground cover imagery (Landsat TM/ETM+/OLI, DEA/JRSRP fractional cover):
1. **Temporal Minimum Ground Cover as a Persistence Indicator**:
   - Computes a temporal minimum ground cover image (or long-term bare ground maximum) across multi-decadal imagery to identify locations retaining cover during climatic stress.
2. **Moving Window Local Neighbourhood**:
   - Evaluates a large moving spatial window (operational default: `1501 × 1501` pixels / ~45 km × 45 km at 30 m resolution) to define a local reference population subject to equivalent macro-climatic conditions.
3. **Upper-Percentile Reference Benchmarks**:
   - Within each moving window (and matching IBRA region when enabled), selects persistent reference pixels in the upper quantile tail (typically the `90th` to `95th` percentiles) and averages their seasonal ground cover.
4. **Difference Ground Cover (DGC)**:
   - Evaluates the central pixel's seasonal ground cover relative to the surrounding reference mean:

     ```math
     \text{DGC} = \text{Groundcover}_{\text{pixel}} - \text{Groundcover}_{\text{reference}}
     ```

### C. Primary Reference and Citation

> **Bastin, G., Scarth, P., Chewings, V., Sparrow, A., Denham, R., Schmidt, M., O'Reagain, P., Shepherd, R., & Abbott, B. (2012).**  
> *Separating grazing and rainfall effects at regional scale using remote sensing imagery: A dynamic reference-cover method.*  
> **Remote Sensing of Environment**, 121, 443–457.  
> DOI: [10.1016/j.rse.2012.02.021](https://doi.org/10.1016/j.rse.2012.02.021)

```bibtex
@article{bastin2012separating,
  title={Separating grazing and rainfall effects at regional scale using remote sensing imagery: A dynamic reference-cover method},
  author={Bastin, G. and Scarth, P. and Chewings, V. and Sparrow, A. and Denham, R. and Schmidt, M. and O'Reagain, P. and Shepherd, R. and Abbott, B.},
  journal={Remote Sensing of Environment},
  volume={121},
  pages={443--457},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.rse.2012.02.021}
}
```

- **Original JRSRP GitLab Repository**: [https://gitlab.com/jrsrp/themes/cover/groundcover/drcm](https://gitlab.com/jrsrp/themes/cover/groundcover/drcm)

---

## 2. Pipeline Architecture & Methodology

```text
[NCAS Woody Extent]      [Terrain Slope]       [Long-Term Bare Stats]      [IBRA 7.0 Vector]
woodyMaxMinCleared.tif  slope_relief_3s.tif    dixa2/dp8a2 bareStats       ibra.fgb
         │                      │                        │                         │
         └──────────────────────┼────────────────────────┴─────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Stage 1: maskChangeSteepIbra.py (RIOS)                   │
         │ - Excludes NCAS Forest (Woody Max == 2)                  │
         │ - Excludes Historical Clearing (Cleared == 1)            │
         │ - Excludes Steep Terrain (Slope relief >= 39 / >= 10%)   │
         │ - Burns IBRA Bioregions (1..255 into Band 2)             │
         └──────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
         2-Band Reference GeoTIFF (Band 1: Masked Stats 1..101; Band 2: IBRA 1..255)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │                                                               │
    ▼ (--ibra, default)                                             ▼ (--no-ibra)
[refpix_numba_ibra.py]                                   [refpix_numba.py]
Serpentine 2D Tiled Regional Kernels                     Serpentine 2D Tiled Global Kernel
- 1..255 regions, k=1 fast path & k>=2 tight bboxes      - Single 104-bin cache-aligned histogram
- 2-bit 256x256 block mask skipping                      - Leading/trailing ocean row trimming
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Stage 3: seasonalDRCMIbra.py                             │
         │ - Auto-detects dp8a2 (0..100%) vs dixa2 (100..200)       │
         │ - Validates aligned grids & zero-pads off-grid extents   │
         │ - Halo-padded --bbox / --geojson & native c_start/c_stop │
         │ - Chunk overlap reuse + background GDAL prefetch         │
         │ - Tiled DEFLATE (PREDICTOR=2) GeoTIFF + overviews        │
         └──────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
         Output GeoTIFF: Band 1 DGC (100 + gc - ref, 0=NoData)
                         Band 2 Reference Cover (100 + ref, 0=NoData)
```

### Technical Rationale: Masking & IBRA Stratification
- **NCAS Woody Extent (`inputs.changeData[0] != 2`)**: Excludes forest (`Class 2`) where tree crowns and shadow confound spectral unmixing of understorey pasture cover.
- **Historical Clearing (`inputs.changeData[2] != 1`)**: Excludes cleared tracts so disturbed sites do not depress reference benchmarks.
- **Terrain Slope Masking (`inputs.terrainData[0] < 39`)**: Retains plains, rises, and low hills (`< 10%` slope per Speight 2009 Table 5) while excluding rolling/steep terrain (`>= 39`).
- **IBRA Stratification vs. `--no-ibra`**:
  - Stratified mode (default in `seasonalDRCMIbra.py`) constrains each pixel's `1501 × 1501` candidate reference window to pixels sharing the exact same `uint8` IBRA region ID (`1..255`, covering IBRA 7.0 regions `1..89`), eliminating cross-bioregion contamination (e.g. floodplains bias-inflating nearby sand dunes).
  - Unconstrained global mode (`--no-ibra`) pools all valid reference pixels in the comparison window.

---

## 3. Numba & Pipeline Optimizations

Both `refpix_numba.py` and `refpix_numba_ibra.py` implement the following optimizations while maintaining bit-for-bit mathematical equivalence with the reference sliding-window definition:

1. **2D Tiled Column Histograms & Automatic Accumulator Tiers**:
   - Decomposes each strip into `col_tile_w=512` 2D tiles with column-major pre-aggregated histograms (`col_perc`, `col_dryhist`, `col_drysum`), avoiding repeated inner-loop scans over columns.
   - Pads histogram width to `104` bins for SIMD/cache-line alignment and uses branchless `_bounds_fast` with pre-tracked non-zero reference totals (`perc_tot`).
   - Automatically selects accumulator widths based on `window`:
     * `window <= 4611`: `uint16` histograms + `int32` ground sums (`_KERNEL_U16` / `_IBRA_KERNEL_U16`) for half the L2 cache footprint.
     * `4611 < window <= 65535`: `uint16` histograms + `int64` sums (`_KERNEL_U16_I64` / `_IBRA_KERNEL_U16_I64`) to prevent `int32` sum overflow when `window^2 * 101 > 2^31 - 1`.
     * `window > 65535`: `int32` histograms + `int64` sums (`_KERNEL_I32` / `_IBRA_KERNEL_I32`).
2. **Serpentine (Boustrophedon) Sweep & `strip_h=16384` Tiling**:
   - Standard row-by-row 2D tiling resets and re-sums all `window` (`1,501`) column histograms whenever stepping from `(r, c1 - 1)` back to `(r + 1, c0)`.
   - Alternating sweep directions—left-to-right on even rows and right-to-left on odd rows—keeps the active `1,501`-column histogram parked directly at the turnaround column, requiring only single-row add/remove updates (`r + halfwindow` and `r - 1 - halfwindow`) at row transitions:

     ```text
     Row r   (even):  c0 ────────────────────────► c1-1   (slide window right)
                                                     │     turnaround: update 1 top + 1 bottom row
     Row r+1 (odd):   c0 ◄──────────────────────── c1-1   (slide window left, zero rebuild)
     ```

   - Raises default strip height to `strip_h=16384` (adaptively halving down toward `128` on smaller rasters/AOIs so all CPU cores stay busy).
3. **Unified Multi-IBRA Specialization (`1..255` Regions)**:
   - Supports all `1..255` `uint8` region IDs (lifted from the legacy `< 90` cap).
   - Uses a 2-bit `256 × 256` block index (`_build_ibra_nonzero_blocks`: bit `0` = `ibra > 0`, bit `1` = valid paired `(cover, ground)`) to skip pure ocean and empty-halo tiles (`_tile_active_core_only`, `_tile_has_valid_halo`).
   - `_scan_core_regions_2d` classifies each tile: single-region interior tiles (`k == 1`) run a zero-branch Serpentine fast path, while boundary tiles (`k >= 2`) run passes clipped strictly to each active region's tight 2D bounding box `(reg_rmin..reg_rmax, reg_cmin..reg_cmax)`.
4. **Native Halo Slicing & Bounds Safety**:
   - Both kernels accept `r_start`, `r_stop`, `c_start`, `c_stop` natively so halo-padded AOI strips never waste compute on padding margins or require post-compute slicing, and enforce strict `0 < val <= 101` range guards.
5. **Sparse-Reference Cloud Recovery & Sample Floor**:
   - Standard window validity passes when `n_valid >= minvalpercent * n_domain` (`--minvalpercent`, default `0.15`).
   - When `--min-obs-percent > 0` (CLI default `0.50`), `_window_passes_gate` also recovers clear pixels inside sparse static-reference landscapes if seasonal cloud-free observability `n_valid >= min_obs_percent * n_pop` and static reference density `n_pop >= sparse_min_valpercent * n_domain` (`--sparse-min-valpercent`, CLI default `0.02`).
   - Enforces a minimum upper-quantile sample floor (`--min-ref-samples`, CLI default `500`; Python API default `0`) inside `_calcmean`, and returns `-1` (`uint8(0)` NoData) if the `[lowerquantile, upperquantile]` band contains zero cloud-free observations (`nsum == 0`).
6. **Pipeline I/O, Stage-1 & Encoding Optimizations**:
   - `seasonalDRCMIbra.py` validates pixel size, axis alignment, and CRS (`_validate_aligned_grid`), cleanly zero-pads off-grid extents (`_read_clipped_band_window`), supports separate `--ibra-raster`, reuses chunk overlap rows (`prev_state`), prefetches the next chunk in a single background I/O thread (`ThreadPoolExecutor(max_workers=1)`), detects Linux cgroup v1/v2 CPU quotas (`_get_effective_cpus`), and runs Numba parallel groundcover LUT conversion and output encoding.
   - `maskChangeSteepIbra.py` uses an ocean block skip, a reusable in-memory GDAL rasterization target, NumPy-2 NEP-50 safe `int16` arithmetic with invalid-value pre-masking, preserves Band 2 IBRA codes across `bareStats == 0` land holes, and performs atomic `.tmp.fgb` `ensure_ibra_albers` reprojection.

### Measured Kernel Benchmark (`2000 × 2000`, `window = 1501`, `4` CPU cores)

Measured on a 4-core `AMD EPYC 9R14` host (`NumPy 2.0.1`, `Numba 0.60.0`, C-contiguous `int8`/`uint8` arrays with `150` leading ocean/NoData rows, warm JIT, verified `0` mismatches against `main` across all 12M output pixels):

| Kernel Mode | `main` Baseline (`e629c6e`) | Serpentine 2D Tiled (`strip_h=16384`) | Speedup |
| :--- | :--- | :--- | :--- |
| **`--no-ibra` (`refpixelfilternodata`)** | `4.935 s` (`0.81M px/s`) | **`0.370 s` (`10.81M px/s`)** | **`13.3×`** |
| **`--ibra` single region (`k = 1`)** | `9.715 s` (`0.41M px/s`) | **`0.378 s` (`10.59M px/s`)** | **`25.7×`** |
| **`--ibra` 4-region boundary (`k = 4`)** | `12.721 s` (`0.31M px/s`) | **`0.272 s` (`14.71M px/s`)** | **`46.8×`** |

---

## 4. Installation & Quick Start (Tested on `test_data/`)

The repository includes ~`495 KiB` of real Australian Albers (`EPSG:3577`) Landsat test chips in `test_data/` so anyone can verify and run the full pipeline locally:
- `test_data/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif` (`520 × 520` — Band 1: Masked reference stats, Band 2: IBRA regions `17` & `58`)
- `test_data/lztmre_aus_m202606202608_dp8a2_vinterim.tif` (`440 × 440` — seasonal bare cover aligned at offset `col=40, row=40` inside the reference grid)
- `test_data/aoi_nsw_ibra_boundary.geojson` (`EPSG:4326` AOI spanning the IBRA 17 / 58 boundary)

### Create the Environment

```bash
conda env create -f environment.yml
conda activate drcm-numba
```

*(Or with Mamba: `mamba env create -f environment.yml && mamba activate drcm-numba`. If user-site packages in `~/.local` shadow Conda's NumPy version and cause a Numba import error, run with `PYTHONNOUSERSITE=1 python ...` or `python -s ...`.)*

### Run the Verification Suite

```bash
python test_ibra_drcm.py
```

### Run `seasonalDRCMIbra.py` on the Included Test Chips

```bash
# 1. Full 440x440 test chip in IBRA-constrained mode (small window=61; --min-ref-samples 0 for small windows)
python seasonalDRCMIbra.py \
    test_data/lztmre_aus_m202606202608_dp8a2_vinterim.tif \
    --percentile-file test_data/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif \
    --window 61 \
    --min-ref-samples 0 \
    -o /tmp/drcm_chip_ibra.tif

# 2. Subset to an EPSG:4326 GeoJSON AOI with halo padding in unconstrained mode (--no-ibra)
python seasonalDRCMIbra.py \
    test_data/lztmre_aus_m202606202608_dp8a2_vinterim.tif \
    --percentile-file test_data/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif \
    --geojson test_data/aoi_nsw_ibra_boundary.geojson \
    --window 61 \
    --min-ref-samples 0 \
    --no-ibra \
    -o /tmp/drcm_aoi_no_ibra.tif
```

*(Note: For operational `1501 × 1501` windows, omit `--window 61 --min-ref-samples 0` to use the operational defaults `--window 1501`, `--minvalpercent 0.15`, `--min-obs-percent 0.50`, `--sparse-min-valpercent 0.02`, `--min-ref-samples 500`.)*

---

## 5. Verification Suite (`test_ibra_drcm.py`)

| Test | Scope | Verified Properties |
| :--- | :--- | :--- |
| **Test 1** | Utility Helpers | `format_time`, `derive_output_name` (`dp8a2`/`dixa2` -> `doda2`/`doda2_ibra` and custom basenames) |
| **Test 2** | Groundcover Encoding Conversion | `convert_to_groundcover` (`dp8a2`, `dixa2`, filename stage inference, fallback, `inplace=True`, empty arrays) |
| **Test 3** | Kernel Micro-Functions | `_bounds`, `_bounds_fast`, multi-bin `_calcmean` `int64` floor rounding, `min_samples`, empty `-1` sentinel |
| **Test 4** | Independent Brute-Force Oracle & Tiled Kernels | Histogram-free sorted-array oracle (`_oracle_refpixel`), `190 × 215` single-region invariance, slicing, `_KERNEL_I32` |
| **Test 5** | Row & Column Boundary Isolation | Zero cross-IBRA contamination across vertical and horizontal bioregion borders |
| **Test 6** | Ocean Masking & Block Skipping | `_build_valid_blocks` empty-block skip parity vs unskipped kernel & oracle, ocean `-1`, `minvalpercent` |
| **Test 7** | Portable Real Satellite Multi-IBRA | `test_data/` `440 × 440` chip offset `(40, 40)` inside `520 × 520` ref, 4 overviews, IBRA 17 vs 58 divergence, oracle parity |
| **Test 8** | Portable Real Satellite `--no-ibra` & `--geojson` | `test_data/` `--no-ibra` oracle parity + `EPSG:4326` `aoi_nsw_ibra_boundary.geojson` (`282 × 282`, 0 mismatches) |
| **Test 9** | GeoTIFF Formats, Prefetch & CLI | `PREDICTOR=1/2`, `INTERLEAVE=BAND/PIXEL`, `COMPRESS=LZW`, `_LocalBandReader`, multi-chunk overlap + `main()` CLI output derivation |
| **Test 10** | Multi-Chunk Overlap & Strict AOI Bounds | Multi-chunk vs single-chunk identity, `(40, 35)` offset, unpadded edge mismatch check, `--bbox`/`--geojson` subset bounds |
| **Test 11** | Serpentine Kernels (`1..255`) & Edge Cases | Serpentine IBRA (`90..255`, `k=1`/`k>=2`, `_IBRA_KERNEL_I32`), `--no-ibra` (`strip_h=16384`), `cover/ground > 101`, `window=4621` `int64` overflow guard, `window=1`, `nrow=1` |
| **Test 12** | `dixa2`, Off-Grid & Stage-1 Pipeline | `dixa2` with `255` chunks, off-grid 4-border zero-padding & `--ibra-raster`, CRS/resolution errors, `maskChangeSteepIbra` underflow/land-hole/attr `> 255` guards, `ensure_ibra_albers` reprojection/cache |
| **Test 13** | Sparse-Ref Recovery, Sample Floor & Empty-Tail NoData | Cloud-free sparse recovery, cloudy fragment rejection, `min_ref_samples` cutoff, empty `[P90, P95]` `uint8(0)` NoData, CLI flags & fuzz parity vs `_oracle_refpixel` |

---

## 6. Scripts and Command-Line Reference

### Script Summary

| Script | Purpose |
| :--- | :--- |
| `maskChangeSteepIbra.py` | RIOS Stage-1 preprocessor generating 2-band masked reference stats (Band 1) and IBRA region codes (Band 2, `1..255`). |
| `refpix_numba_ibra.py` | Multi-threaded Serpentine 2D tiled Numba kernel (`--ibra`, regions `1..255`). |
| `refpix_numba.py` | Multi-threaded Serpentine 2D tiled unconstrained Numba kernel (`--no-ibra`). |
| `seasonalDRCMIbra.py` | Operational CLI driver for seasonal DRCM processing with chunk overlap reuse, background GDAL prefetch, AOI subsetting, and GeoTIFF output + overviews. |
| `test_ibra_drcm.py` | Standalone 13-test verification suite using an independent oracle and `test_data/`. |
| `environment.yml` | Minimal `conda-forge` environment definition (`python`, `numpy`, `numba`, `gdal`, `rios`). |

### Stage 1: Generate the 2-Band Reference + IBRA Raster

```bash
python maskChangeSteepIbra.py \
    --change woodyMaxMinCleared.tif \
    --terrain slope_relief_class_3s.tif \
    --bare lztmre_aus_m198912202012_dixa2_bareStats.tif \
    --ibra ibra.fgb \
    -o lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif
```

- Automatically reprojects vector boundaries to match the reference raster CRS (`ensure_ibra_albers` -> `*_albers.fgb`) using atomic file replacement.
- Supports `--bare-encoding {auto,dixa2,dp8a2}`, `--ibra-attr <FIELD>` (default `OBJECTID`, validated in `1..255`), and `--test-chip XOFF YOFF W H`.

### Stage 2/3: Run Seasonal DRCM (`seasonalDRCMIbra.py`)

```bash
python seasonalDRCMIbra.py lztmre_aus_m202606202608_dp8a2_vinterim.tif \
    --percentile-file lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif \
    --window 1501 \
    --chunksize 4096
```

#### Key CLI Flags
- `input_groundcover`: Seasonal bare ground mosaic (`dp8a2` `0..100%` or `dixa2` `100..200`).
- `--percentile-file`: 2-band reference GeoTIFF from Stage 1 (or 1-band reference paired with `--ibra-raster` or `--no-ibra`).
- `--ibra-raster`: Optional separate IBRA raster when `--percentile-file` is single-band.
- `-o, --output`: Output GeoTIFF path (or directory ending with `/`); derived automatically if omitted (`*_doda2_ibra*.tif` or `*_doda2*.tif`).
- `--window`: Odd comparison window size in pixels (default: `1501`).
- `--chunksize`: Vertical chunk height in rows (default: `4096`).
- `--lowerquantile` / `--upperquantile`: Reference percentile bounds (defaults: `0.90` / `0.95`).
- `--minvalpercent`: Minimum valid paired pixel fraction in window (default: `0.15`).
- `--min-obs-percent`: Minimum seasonal cloud-free fraction of static reference pixels for sparse-reference recovery (`0` disables; CLI default: `0.50`).
- `--sparse-min-valpercent`: Minimum static reference density eligible for sparse-reference recovery (CLI default: `0.02`).
- `--min-ref-samples`: Minimum seasonal observations required inside `[lowerquantile, upperquantile]` (CLI default: `500`).
- `--bbox MINX MINY MAXX MAXY` / `--geojson PATH`: Halo-padded spatial AOI subsetting matching full-extent outputs.
- `--test-chip XOFF YOFF W H`: Unpadded micro-subsetting window for prototyping.
- `--no-ibra`: Disable IBRA stratification and use the unconstrained global histogram kernel.
- `--bare-encoding {auto,dp8a2,dixa2}`: Input bare ground scaling convention (default: `auto`).
- `--threads N`: Numba / GeoTIFF thread count (default: detected CPUs / cgroup quota).
- `--overviews ...` / `--no-overviews`: Overview levels (default: `4 8 16 32`) or skip overview generation.

#### Output Product Specification (`GTiff`)
- **Compression & Layout**: `COMPRESS=DEFLATE`, `ZLEVEL=1`, `PREDICTOR=2`, `TILED=YES`, `INTERLEAVE=BAND`, `BLOCKXSIZE=512`, `BLOCKYSIZE=512`, `BIGTIFF=YES`.
- **Overviews**: Internal `DEFLATE` overviews (`PREDICTOR_OVERVIEW=1`) at levels `4, 8, 16, 32` using `AVERAGE` resampling.
- **Band 1 — Difference Ground Cover (`DGC`)**: `uint8`, `100 + (Groundcover - Refpixel)`, `0 = NoData` (`100` = zero difference, `> 100` = surplus cover, `< 100` = deficit cover).
- **Band 2 — Reference Ground Cover**: `uint8`, `100 + Refpixel` (`100 = 0%`, `200 = 100%`), `0 = NoData`.
