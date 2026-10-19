# Dynamic Reference Cover Method (DRCM) Pipeline

This repository contains a high-performance Python and Numba implementation of the Australian seasonal Groundcover Reference Pixel Filter and DRCM processing pipeline. The software applies National Carbon Accounting System (NCAS) woody extent masks and Interim Biogeographic Regionalisation for Australia (IBRA) bioregion boundaries.

This software replaces the 15-year-old Fortran 90 module (`refpixelfilter.f90`) and legacy scripts (`seasonalDCRM.py`, `maskChangeSteep.py`). It provides memory-safe processing, parallel multi-threading, and verified numerical accuracy.

---

## 1. System Architecture and Workflow

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
         │ - Excludes Steep Terrain (Slope relief >= 39 / >= 10%)  │
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
         │ - Calculates internal overviews (4, 8, 16, 32)          │
         └──────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
     Final DRCM GeoTIFF: doda2_ibra_vinterim.tif
     Band 1: Difference Ground Cover (DGC, IBRA-Constrained)
     Band 2: Reference Ground Cover (IBRA-Constrained)
```

---

## 2. Technical Rationale: NCAS Forest Masking and IBRA Stratification

### A. NCAS Woody Extent and Terrain Slope Masking
* **Purpose**: DRCM measures rangeland pasture condition (grasses, forbs, and litter) relative to local persistent reference sites. In dense forest environments (tree canopy cover > 20%), spectral unmixing does not measure groundcover accurately because tree crowns cause dense shadow and light scattering.
* **Filter Rules**:
  - `inputs.changeData[0] != 2`: Band 1 of `woodyMaxMinCleared.tif` contains NCAS Woody Maximum (0 = Non-woody, 1 = Woodland, 2 = Forest). The software excludes pixels with forest cover (Class 2). This makes sure that reference sites represent open rangelands and grazing pasture.
  - `inputs.changeData[2] != 1`: Band 3 records historical clearing events (1 = Cleared). The software excludes cleared land to prevent disturbed tracts from lowering baseline standards.
  - `inputs.terrainData[0] < 39`: Based on Speight (2009) Table 5 (CSIRO and Geoscience Australia landform pattern classification in `dem_s_slope_relief_3s_metadata.doc`). The two-digit values combine median slope (tens digit) and relief (units digit):
    - Tens digit (Median slope): `10` (< 1%, level), `20` (1-3%, very gently sloping), `30` (3-10%, gently sloping), `40` (10-32%, moderately sloping / rolling), `50` (32-56%, steep), `60` (56-100%, very steep), `70` (> 100%, precipitous).
    - Codes < 39 select only plains, rises, and low hills with median slope < 10%.
    - Codes >= 39 exclude all rolling, steep, and mountainous terrain (slope >= 10%) and badlands. This eliminates areas where hillshade and terrain shadow distort pasture unmixing.

### B. IBRA Biogeographic Region Stratification
* **Purpose**: The operational comparison window is 1501 x 1501 pixels (45 km x 45 km, ~2,000 km2). Across biogeographic boundaries, this window can cover different environments, such as dry sand dunes next to river floodplains.
* **The Problem**: An unconstrained window selects high cover values from the river floodplain to evaluate the sand dunes. This causes false degradation anomalies in the sand dunes.
* **The Solution**: The software uses `ibra.fgb` (89 continental IBRA 7.0 regions). Each target pixel samples only reference sites from the same matching IBRA region.

---

## 3. High-Performance Multi-IBRA Engine: `refpix_numba_ibra.py`

### Memory Architecture and L2 Cache Locality
Instead of a single 1D histogram, the multi-IBRA engine maintains a 2D table for each thread:
- `percentilehist = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `dryhist = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `drysum = np.zeros((90, 103), dtype=np.int32)` (37 KB)
- `valid_counts = np.zeros(90, dtype=np.int32)` (360 bytes)
- `total_counts = np.zeros(90, dtype=np.int32)` (360 bytes)

**Total memory per thread: ~112 KB**.
This buffer fits in the private L2 cache (512 KB to 1 MB) of each CPU core. This design avoids memory bus delays during row traversal.

### O(1) Sliding Window Updates
When moving one column to the right:
1. The software reads pixels in the new column sliver:
   `reg = ibra[k, right_c]`
   If `reg > 0`, it updates `percentilehist[reg, pixval] += 1` and `dryhist[reg, pixval] += 1`.
2. The software subtracts pixels in the old column sliver:
   `reg = ibra[k, left_c]`
   If `reg > 0`, it updates `percentilehist[reg, pixval] -= 1`.
3. **Zero Looping Across Regions**: Updating the window requires O(window_height) operations per pixel. It does not loop through the 89 regions.
4. **Target Evaluation**: The algorithm evaluates only the target region: `target_reg = ibra[r, c]`.

---

## 4. Modernization and Performance Benchmarks

### A. Deployment and Toolchain
- **Python >= 3.12 Compatibility**: Does not require legacy `distutils`, `f2py`, `gfortran`, `meson`, or `ninja` compilers at runtime.
- **Pure Python Delivery**: Runs natively via Numba JIT on standard Python and NumPy environments.

### B. Performance Benchmarks
Measured on 1000 x 1000 to 2000 x 2000 satellite arrays (4,000,000 pixels, comparison window = 1501):

| Implementation | Runtime (2000 x 2000) | Throughput | Speedup vs Fortran |
| :--- | :--- | :--- | :--- |
| **Fortran 90 (`f2py` single-thread)** | 12.38 s | 323,141 px/s | 1.0x (Baseline) |
| **Numba Serial (1 core)** | 26.40 s | 151,500 px/s | 0.47x |
| **Numba Parallel (`prange`, 8 cores)** | **5.80 s** | **689,599 px/s** | **2.13x faster** |

*For medium windows (`window=51`), Numba parallel achieves > 12,000,000 px/s (1.72x faster than Fortran).*

### C. Block Rasterization Optimizations in `maskChangeSteepIbra.py`
- **Ocean Fast-Path Exit**: Ocean covers > 60% of the Australian bounding box. When a block contains no valid bare ground data (`not np.any(bareStats > 0)`), the script exits in 0.005 ms. This skips > 45,000 ocean blocks.
- **Reusable Memory Dataset**: The script creates one in-memory GDAL dataset in `main()` and updates it per block. This removes 73,128 dataset creations and deletions.
- **Single-Pass Spatial Queries**: Removed redundant `GetFeatureCount()` and `SetSpatialFilter(None)` calls.
- **Speedup**: Processing time decreased from 64.5 ms to 30.2 ms per block (2.13x faster), saving ~42 minutes across Australia.

---

## 5. Verification and Test Suite (`test_ibra_drcm.py`)

The test suite in `test_ibra_drcm.py` provides 100% test coverage across all micro-functions, algorithms, and integration drivers:

### Test 1: Utility Functions (`format_time`, `derive_output_name`)
- Validates duration formatting across boundary values (`0s`, `59s`, `60s`, `125s`, `3599s`, `3600s`, `3665s`).
- Validates output filename derivation from `dixa2`, `dp8a2`, and generic basenames.

### Test 2: Groundcover Encoding Conversion (`convert_to_groundcover`)
- Validates modern DEA (`dp8a2`) unscaled percentage conversion (`0..100%` bare with `255` NoData mapped to `101..1` groundcover and `0` NoData).
- Validates legacy JRSRP (`dixa2`) 100-offset conversion (`100..200` bare with `0` NoData mapped to `101..1` groundcover and `0` NoData).

### Test 3: Kernel Functions (`_bounds`, `_calcmean`)
- Validates quantile bounds calculation on discrete and degenerate distributions (`b0=79, b1=79` for 90-95th; `b0=49, b1=79` for 40-60th).
- Validates mean groundcover calculation within bounds.
- Confirms safe handling of empty histograms (returns 0 without division-by-zero errors).

### Test 4: Vector Reprojection and Caching (`ensure_ibra_albers`)
- Validates OGR vector reprojection to Australian Albers (`ibra_albers.fgb`).
- Confirms metric coordinates (> 100,000 m), 89-feature preservation, and sub-millisecond cached re-reads.

### Test 5: Fortran 90 Bit-for-Bit Verification (`refpix_numba` vs compiled Fortran)
- Tested on 40,000 pixels comparing `refpix_numba.py` directly against compiled Fortran `refpixelfilter.so`.
- **Result: 0 mismatches (100% bit-for-bit identical)**.

### Test 6: Single-Region Invariance (`refpix_numba_ibra` vs `refpix_numba`)
- Tested on 40,000 pixels inside a single IBRA region comparing the unconstrained kernel against the multi-IBRA kernel.
- **Result: 0 mismatches (100% bit-for-bit identical)**.

### Test 7: Cheat-Proof Boundary Isolation
- Synthetic 100 x 100 grid split at column 50:
  - Left (col < 50): Region 10 (High cover: 95/80 ref, 90/70 ground)
  - Right (col >= 50): Region 20 (Low cover: 35/20 ref, 30/15 ground)
  - Window = 21 (overlapping boundary by +/- 10 pixels).
- **Results**:
  - Center pixel in Region 10 (col 49): strictly outputs **90** (local ground is 70).
  - Center pixel in Region 20 (col 50): strictly outputs **30** (local ground is 15).
  - Global unconstrained filter at col 50: outputs **72** (contaminated by Region 10 across boundary).
  - **Result: Zero cross-boundary contamination**.

### Test 8: Ocean Masking and Validity Thresholds
- Confirmed that pixels with `ibra == 0` (ocean or external bounding box) are strictly masked to `-1` (NoData).
- Confirmed that pixels below the valid fraction threshold (`minvalpercent`) output `-1`, while pixels above the threshold output valid data.

### Test 9: Real Satellite Chip Multi-IBRA End-to-End Test
- Extracted 2,250,000 pixels from NSW mosaic (`lztmre_nsw_...`) across the boundary of Darling Riverine Plains (Region 17) and Cobar Peneplain (Region 58).
- Runs `maskChangeSteepIbra.py` followed by `seasonalDRCMIbra.py`.
- **Results**:
  - Output contains > 2,100,000 valid pixels.
  - Confirmed that mean reference cover differs between Region 17 and Region 58.
  - Confirmed internal overviews exist for levels 4, 8, 16, 32.

### Test 10: Single Global Histogram Mode (`--no-ibra`)
- Evaluates `seasonalDRCMIbra.py` with `--no-ibra` (single global histogram mode) against the direct unconstrained kernel.
- Confirms output band descriptions update to `Unconstrained`.
- **Result: 0 mismatches (100% bit-for-bit exact match)**.

---

## 6. Scripts and Command-Line Usage

### Script Summary

| Script | Purpose |
| :--- | :--- |
| `maskChangeSteepIbra.py` | Generates 2-band masked baseline stats (Band 1) and IBRA region codes (Band 2). |
| `refpix_numba_ibra.py` | Multi-threaded Numba kernel holding 90 regional histograms. |
| `refpix_numba.py` | Standalone unconstrained single-histogram Numba kernel. |
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

- Ingests `ibra.fgb` and reprojects it to Australian Albers (`ibra_albers.fgb`).
- Uses RIOS standard `cuiprogress.CUIProgressBar` for terminal progress display.
- Uses block-level in-memory rasterization (`gdal.RasterizeLayer` with `SetSpatialFilterRect`), using 0 MB of temporary disk space in `/tmp`.

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
- **Compression**: `COMPRESS=DEFLATE`, `ZLEVEL=1`, `PREDICTOR=1`, `TILED=YES`, `BLOCKXSIZE=512`, `BLOCKYSIZE=512`, `BIGTIFF=YES`.
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
Validates single-region invariance, boundary isolation, ocean masking, validity thresholds, overviews, and satellite multi-IBRA execution.
