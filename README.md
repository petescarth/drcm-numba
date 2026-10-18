# Modernized Deficit / Reference Cover Method (DRCM) Pipeline

High-performance, pure-Python / Numba implementation of the Australian seasonal Groundcover Reference Pixel Filter and DRCM processing pipeline, featuring NCAS Woody Extent filtering and IBRA biogeographic region stratification.

Replaces the legacy 15-year-old Fortran 90 (`refpixelfilter.f90`) and Python 2 / RSC scripts (`seasonalDCRM.py`, `maskChangeSteep.py`) with memory-safe, multi-threaded, and scientifically refined workflows.

---

## 1. Architectural Overview & Workflow

```
[NCAS Woody Extent]      [Terrain Slope]       [1989-2020 Bare Stats]      [IBRA 7.0 Vector]
woodyMaxMinCleared.tif  slope_relief_3s.tif   dixa2_bareStats.tif (Band 7)      ibra.fgb
         │                      │                        │                         │
         └──────────────────────┼────────────────────────┴─────────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────────────────┐
         │ Stage 1: maskChangeSteepIbra.py (RIOS)                   │
         │ - Excludes NCAS Forest (Woody Max == 2)                 │
         │ - Excludes Historical Clearing (Cleared == 1)           │
         │ - Excludes Steep Terrain (Slope relief >= 39 / >40%)    │
         │ - Burns IBRA Bioregions (OBJECTID 1..89 into Band 2)    │
         └──────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     Output Reference GeoTIFF: dixa2_change_slope2_ibra.tif
     Band 1: Masked Baseline Groundcover (1..101, 0=NoData)
     Band 2: IBRA Biogeographic Region ID (1..89, 0=NoData)
                                    │
    ┌───────────────────────────────┴──────────────────────────────┐
    │                                                              │
    ▼                                                              ▼
[Seasonal Fractional Cover]                         [Stage 2: refpix_numba_ibra.py]
dp8a2 (DEA) or dixa2 (JRSRP)                        Parallel Multi-IBRA Sliding Window
    │                                               - 90 Independent Regional Histograms
    │                                               - O(1) Cache-Aligned Updates
    └───────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
         ┌─────────────────────────────────────────────────────────┐
         │ Stage 3: seasonalDRCMIbra.py                            │
         │ - Dynamic GeoTransform Spatial Alignment                │
         │ - Auto-detects dp8a2 vs dixa2 encodings                 │
         │ - Memory-bounded chunk processing with live ETA         │
         └──────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     Final DRCM GeoTIFF: doda2_ibra_vinterim.tif
     Band 1: Difference Ground Cover (DGC, IBRA-Constrained)
     Band 2: Reference Ground Cover (IBRA-Constrained)
```

---

## 2. Scientific Rationale: NCAS Forest Masking & IBRA Stratification

### A. NCAS Woody Extent & Clearing Masking
* **Core Problem**: DRCM quantifies rangeland groundcover (pastoral grasses, forbs, and litter) relative to local persistent reference condition. In dense climax forest environments (canopy cover > 20%), spectral unmixing of fractional groundcover breaks down due to dense canopy shadow, multi-layer scattering, and crown interception.
* **The Solution**: 
  - `inputs.changeData[0] != 2`: Band 1 of `woodyMaxMinCleared.tif` contains NCAS Woody Maximum (0 = Non-woody, 1 = Woodland, 2 = Forest). Pixels that were ever dense closed forest are masked out.
  - `inputs.changeData[2] != 1`: Band 3 tracks historical clearing events (1 = Cleared). Cleared tracts are masked to prevent anthropogenically disturbed sites from artificially lowering reference standards.
  - `inputs.terrainData[0] < 39`: Based on the CSIRO/Geoscience Australia Slope-Relief classification (Gallant & Wilford / Speight 2009, Table 5 in `dem_s_slope_relief_3s_metadata.doc`), the raster codes combine median slope (tens digit) and relief (units digit):
    - Tens digit: `10` (<1%, level), `20` (1-3%, very gently sloping), `30` (3-10%, gently sloping), `40` (10-32%, moderately sloping / rolling), `50` (32-56%, steep), `60` (56-100%, very steep), `70` (>100%, precipitous).
    - Codes < 39 include only level plains (`11`), gently inclined plains/rises (`21`, `22`), and undulating plains/rises/low hills (`31`, `32`, `33`, `34`, `35`) where median slope is < 10%.
    - Codes >= 39 (starting at `41`) exclude all moderately sloping, rolling, steep, and mountainous relief (>= 10% median slope) as well as badlands (`51`, `61`, `62`, `63`, `71`, `72`, `73`), eliminating terrain where hillshade, steep shadowing, and rapid runoff distort spectral pasture unmixing.

### B. IBRA Biogeographic Region Stratification
* **Core Problem**: The standard operational comparison window is 1501 x 1501 pixels (45 km x 45 km, ~2,000 km2). Across biogeographic boundaries, a window frequently spans stark ecological discontinuities:
  - An arid dunefield (e.g., Simpson-Strzelecki Dunefields) immediately adjacent to a riverine floodplain (e.g., Darling Riverine Plains).
  - Mulga shrublands abutting basaltic cracking-clay grasslands.
  - When unconstrained, the 90-95th percentile filter in the dunefield searches across the boundary into the riverine flat, sampling unrepresentative fertile alluvial cover (80-90%). This causes massive false "degradation" anomalies in the dunefield.
* **The Solution**:
  - `ibra.fgb` contains the 89 continental biogeographic regions of IBRA 7.0 (`OBJECTID` 1 to 89).
  - `maskChangeSteepIbra.py` burns these codes into **Band 2** of the reference raster.
  - `refpix_numba_ibra.py` partitions comparison window statistics: each target pixel samples **only** reference sites and groundcover from its matching IBRA region.

---

## 3. High-Performance Multi-IBRA Engine: `refpix_numba_ibra.py`

### Memory Architecture & L2 Cache Locality
Instead of a single 1D histogram, the multi-IBRA engine maintains a 2D table per thread:
- `percentilehist = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `dryhist = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `drysum = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `valid_counts = np.zeros(90, dtype=np.int32)` (360 bytes)
- `total_counts = np.zeros(90, dtype=np.int32)` (360 bytes)

**Total memory per thread: ~112 KB**.
Because modern CPU cores provide 512 KB to 1 MB of private L2 cache, these tables stay resident in L2 cache during the entire row traversal, avoiding RAM bus contention.

### O(1) Sliding Window Updates
When advancing one column to the right:
1. For each pixel in the entering right-hand column sliver:
   `reg = ibra[k, right_c]`
   If `reg > 0`, it directly indexes and increments `percentilehist[reg, pixval] += 1` and `dryhist[reg, pixval] += 1`.
2. For each pixel in the leaving left-hand column sliver:
   `reg = ibra[k, left_c]`
   It directly decrements `percentilehist[reg, pixval] -= 1`.
3. **Zero Looping Over Regions**: Neither the entry nor exit step loops through the 89 regions. Updating the window remains O(window_height) per pixel.
4. **Target Evaluation**: The center pixel queries only `target_reg = ibra[r, c]`. Pixels belonging to other IBRA regions inside the window are ignored.

---

## 4. Numba Modernization vs. Legacy Fortran 90

### A. Deployment & Toolchain Modernization
- **Python >= 3.12 Compatibility**: `f2py` deprecated and removed the legacy `distutils` backend in Python 3.12, requiring `gfortran`, `meson`, and `ninja`.
- **Pure Python Delivery**: `refpix_numba_ibra.py` runs natively via Numba JIT on standard Python/NumPy environments without C/Fortran compiler toolchains or binary wheel compilation.

### B. Performance & Profiling Benchmarks
Evaluated on 1000 x 1000 to 2000 x 2000 real satellite chips (4,000,000 pixels, operational `window=1501`):

| Implementation | Runtime (2000 x 2000) | Throughput | Speedup vs Fortran |
| :--- | :--- | :--- | :--- |
| **Fortran 90 (`f2py` single-thread)** | 12.38 s | 323,141 px/s | 1.0x (Baseline) |
| **Numba Serial (1 core)** | 26.40 s | 151,500 px/s | 0.47x |
| **Numba Parallel (`prange`, 8 cores)** | **5.80 s** | **689,599 px/s** | **2.13x faster** |

*On medium windows (`window=51`), Numba parallel achieves > 12,000,000 px/s (1.72x faster than Fortran).*

---

## 5. Verification & Exact Match Testing

The test suite in `test_ibra_drcm.py` provides 100% test coverage across all micro-functions, algorithms, and integration drivers:

### Test 1: Utility Functions (`format_time`, `derive_output_name`)
- Validates human-readable duration formatting (seconds, minutes, and hours).
- Validates output filename derivation from `dixa2`, `dp8a2`, and generic basenames.

### Test 2: Groundcover Encoding Conversion (`convert_to_groundcover`)
- Validates modern DEA (`dp8a2`) unscaled percentage conversion (`0..100%` bare with `255` NoData mapped to `101..1` groundcover and `0` NoData).
- Validates legacy JRSRP (`dixa2`) 100-offset conversion (`100..200` bare with `0` NoData mapped to `101..1` groundcover and `0` NoData).

### Test 3: Kernel Micro-Functions (`_bounds`, `_calcmean`)
- Validates quantile bounds calculation on discrete and degenerate distributions.
- Validates mean groundcover calculation within quantile bounds.
- Confirms safe handling of empty / zero-valid histograms (safely returning 0 without division-by-zero errors).

### Test 4: Vector Reprojection & Caching (`ensure_ibra_albers`)
- Validates OGR vector reprojection to Australian Albers (`ibra_albers.fgb`).
- Confirms 89-feature preservation and instant sub-millisecond cached re-reads.

### Test 5: Fortran 90 Bit-for-Bit Verification (`refpix_numba` vs compiled Fortran)
- Tested on 40,000 pixels comparing `refpix_numba.py` directly against the compiled Fortran `refpixelfilter.so`.
- **Result: 0 mismatches (100% bit-for-bit identical)**.

### Test 6: Single-Region Invariance (`refpix_numba_ibra` vs `refpix_numba`)
- Tested on 40,000 pixels inside a single IBRA region comparing the unconstrained kernel against the multi-IBRA kernel.
- **Result: 0 mismatches (100% bit-for-bit identical)**.

### Test 7: Synthetic Boundary Isolation
- Synthetic 100 x 100 grid split at column 50:
  - Left (col < 50): Region 10 (High cover: 90 ref, 85 ground)
  - Right (col >= 50): Region 20 (Low cover: 30 ref, 25 ground)
  - Window = 21 (overlapping boundary by +/- 10 pixels).
- **Results**:
  - Global filter at column 50: **85** (*contaminated by Region 10 across boundary*).
  - IBRA filter at column 50: **25** (*strictly preserved Region 20 reference*).
  - IBRA filter at column 49: **85** (*strictly preserved Region 10 reference*).
  - **Result: Zero cross-boundary contamination**.

### Test 8: NoData & Ocean Masking
- Confirmed that pixels with `ibra == 0` (ocean or external bounding box) are strictly masked to `-1` (NoData).

### Test 9: Real Satellite Chip Multi-IBRA End-to-End Test
- Extracted 2,250,000 pixels from NSW mosaic (`lztmre_nsw_...`) across the boundary of Darling Riverine Plains (Region 17) and Cobar Peneplain (Region 58).
- Runs `maskChangeSteepIbra.py` followed by `seasonalDRCMIbra.py`.
- **Results**:
  - Executed in 1.6 s (1.41M px/s).
  - All output pixels bounded in [0, 201].
  - Clean spatial demarcation along IBRA vector boundaries.

### Test 10: Single Global Histogram Mode (`--no-ibra`)
- Evaluates `seasonalDRCMIbra.py` with `--no-ibra` (single global histogram mode) against the unconstrained reference pipeline `seasonalDRCM.py`.
- Verifies output band descriptions update to `Unconstrained`.
- **Results**:
  - Diff DGC: **0**
  - Diff Ref Cover: **0**
  - **Result: 100% bit-for-bit exact match with unconstrained baseline**.

---

## 6. Scripts & Command-Line Usage

### Script Summary

| Script | Purpose |
| :--- | :--- |
| `maskChangeSteepIbra.py` | Generates 2-band masked baseline stats (Band 1) and IBRA region codes (Band 2). |
| `refpix_numba_ibra.py` | Multi-threaded Numba kernel holding 90 regional histograms. |
| `seasonalDRCMIbra.py` | CLI driver for processing seasonal groundcover with IBRA constraints or single global histogram (`--no-ibra`). |
| `test_ibra_drcm.py` | Automated test suite validating all 10 verification stages. |

---

### Step 1: Generate the 2-Band Reference Raster

```bash
conda run -n rios python maskChangeSteepIbra.py \
    --change woodyMaxMinCleared.tif \
    --terrain slope_relief_class_3s.tif \
    --bare lztmre_aus_m198912202012_dixa2_bareStats.tif \
    --ibra ibra.fgb \
    -o lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif
```

- Ingests `ibra.fgb` and auto-reprojects it to Australian Albers (`ibra_albers.fgb`).
- Employs RIOS's standard `cuiprogress.CUIProgressBar` for terminal progress display.
- Employs block-level in-memory rasterization (`gdal.RasterizeLayer` with `SetSpatialFilterRect`), using 0 MB of temporary disk space in `/tmp`.

---

### Step 2: Run IBRA-Constrained Seasonal DRCM

```bash
conda run -n rios python seasonalDRCMIbra.py lztmre_nsw_m202606202608_dp8a2_vinterim.tif \
    --percentile-file lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif \
    --window 1501 \
    --chunksize 2048
```

#### Key Parameters:
- `input_groundcover`: Path to seasonal mosaic (`dp8a2` unscaled 0..100% or `dixa2` 100-offset).
- `--percentile-file`: 2-band GeoTIFF from Step 1.
- `--window`: Operational comparison window (default: 1501, ~45 km).
- `--chunksize`: Vertical chunk height in rows (default: 2048, aligned to 4 x 512 tile boundaries to eliminate tile re-reading).
- `--overviews`: Overview levels to calculate using average resampling (default: 4 8 16 32).
- `--no-overviews`: Flag to skip overview generation if desired.
- `--test-chip XOFF YOFF W H`: Micro-subsetting window for fast local prototyping before full continental runs.
- `--no-ibra`: Disables IBRA regional constraints and runs with a single global comparison histogram.
- `--threads`: Number of CPU threads (default: all available cores).

#### Output Product Specification (`GTiff`):
- **Compression**: `COMPRESS=DEFLATE`, `ZLEVEL=1`, `PREDICTOR=2`, `TILED=YES`, `BLOCKXSIZE=512`, `BLOCKYSIZE=512`, `BIGTIFF=YES`.
- **Overviews**: Internal DEFLATE-compressed overviews at 4x, 8x, 16x, 32x downsampling using AVERAGE resampling.
- **Band 1**: `Difference Ground Cover (DGC)`
  - Scaled as: 100 + (Groundcover - Refpixel)
  - 100 = Zero Difference (matches reference)
  - > 100 = Surplus Cover (greener / higher cover than reference)
  - < 100 = Cover Deficit (degraded / lower cover than reference)
  - 0 = NoData
- **Band 2**: `Reference Ground Cover`
  - Scaled as: 100 + Refpixel (100 = 0%, 200 = 100%)
  - 0 = NoData

---

### Step 3: Run the Verification Test Suite

```bash
conda run -n rios python test_ibra_drcm.py
```
Validates single-region invariance, boundary isolation, ocean masking, and satellite chip multi-IBRA execution.
