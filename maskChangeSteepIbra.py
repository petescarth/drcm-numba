#!/usr/bin/env python
"""
Filename:  maskChangeSteepIbra.py
Purpose:   Produces a 2-band masked long-term statistics image for IBRA-constrained DRCM processing:
             - Band 1: Long-term bare ground stats masked by clearing and slope (1..101, 0=nodata)
             - Band 2: Rasterized IBRA biogeographic region code (1..89 from OBJECTID, 0=nodata)
"""

import os
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from rios import applier, cuiprogress

gdal.UseExceptions()


def ensure_ibra_albers(ibra_vector_path, target_proj_wkt, output_albers_path=None):
    """
    Ensure the IBRA vector is reprojected to the raster's projection (Australian Albers)
    for fast spatial filtering and rasterization.
    """
    if output_albers_path is None:
        base, _ = os.path.splitext(ibra_vector_path)
        output_albers_path = f"{base}_albers.fgb"

    if os.path.exists(output_albers_path):
        return output_albers_path

    print(f"Reprojecting {ibra_vector_path} to match reference projection -> {output_albers_path}...")
    opt = gdal.VectorTranslateOptions(dstSRS=target_proj_wkt, format="FlatGeobuf")
    ds = gdal.VectorTranslate(output_albers_path, ibra_vector_path, options=opt)
    ds = None
    return output_albers_path


def maskStatsIbra(info, inputs, outputs, otherArgs):
    """
    RIOS block callback:
      - Band 1: Bare ground 95th/max stats masked by clearing and slope.
      - Band 2: Rasterized IBRA region code (OBJECTID 1..89).
    """
    nullValBare = 0
    (nRows, nCols) = inputs.terrainData.shape[1:]
    outImage = np.zeros((2, nRows, nCols), dtype=np.uint8)

    # Band 1: Bare ground masked by clearing and slope:
    # 1. NCAS Woody Extent: changeData[0] is Woody Max (0=non-woody, 1=woodland, 2=forest).
    #    We exclude Class 2 (forest, canopy cover >20%) to ensure reference sites reflect
    #    open rangeland and grazing pasture rather than closed woody canopy unmixing.
    # 2. Historical Clearing: changeData[2] is clearing history (1=cleared). Excluded to prevent
    #    disturbed / cleared tracts from skewing persistent reference condition.
    # 3. Slope Relief: terrainData[0] < 39 masks terrain based on Speight (2009) Table 5
    #    (CSIRO/GA Slope-Relief). Tens digit is median slope: 10 (<1%), 20 (1-3%), 30 (3-10%),
    #    40 (10-32%), 50 (32-56%), 60 (56-100%), 70 (>100%). Classes < 39 keep only level,
    #    gently inclined, and undulating plains/rises/low hills (<10% slope), excluding
    #    moderately sloping, rolling, steep, mountainous relief (>=10% slope), and badlands.
    # 4. Long-term Bare: bareStats[6] (Band 7) is maximum bare ground across 1989-2020.
    #    Scaled to reference groundcover + 1 (1..101 range, 0 = NoData).
    outImage[0] = np.clip(np.where(
        (inputs.changeData[0] != 2) & # Exclude forest
        (inputs.changeData[2] != 1) & # Exclude historically cleared
        (inputs.terrainData[0] < 39) & # Exclude slope classes >= 40 (median slope >= 10%)
        (inputs.bareStats[6] != nullValBare), # Valid bare stats
        201 - inputs.bareStats[6], 0), 0, 101)

    # Band 2: Rasterize IBRA region code for this exact block in memory
    grid_gt = info.getTransform()
    col_in_grid, row_in_grid = info.getPixColRow(0, 0)
    xRes, yRes = grid_gt[1], grid_gt[5]
    tl_x = grid_gt[0] + col_in_grid * xRes
    tl_y = grid_gt[3] + row_in_grid * yRes
    br_x = tl_x + nCols * xRes
    br_y = tl_y + nRows * yRes

    xMin_b = min(tl_x, br_x)
    xMax_b = max(tl_x, br_x)
    yMin_b = min(tl_y, br_y)
    yMax_b = max(tl_y, br_y)

    ext = otherArgs.extent
    # Check if block intersects IBRA vector bounding box
    if not (xMax_b < ext[0] or xMin_b > ext[1] or yMax_b < ext[2] or yMin_b > ext[3]):
        otherArgs.lyr.SetSpatialFilterRect(xMin_b, yMin_b, xMax_b, yMax_b)
        if otherArgs.lyr.GetFeatureCount() > 0:
            mem_driver = gdal.GetDriverByName("MEM")
            block_mem = mem_driver.Create("", nCols, nRows, 1, gdal.GDT_Byte)
            block_mem.SetProjection(otherArgs.proj)
            block_mem.SetGeoTransform((tl_x, xRes, 0, tl_y, 0, yRes))
            block_mem.GetRasterBand(1).Fill(0)

            gdal.RasterizeLayer(block_mem, [1], otherArgs.lyr, options=["ATTRIBUTE=OBJECTID"])
            outImage[1] = block_mem.GetRasterBand(1).ReadAsArray()
            block_mem = None
        otherArgs.lyr.SetSpatialFilter(None)

    outputs.dcrmMask = outImage


def parse_args():
    parser = argparse.ArgumentParser(description="Create 2-band masked stats + IBRA GeoTIFF for DRCM.")
    parser.add_argument("--change", default="woodyMaxMinCleared.tif", help="Woody change raster")
    parser.add_argument("--terrain", default="slope_relief_class_3s.tif", help="Slope relief raster")
    parser.add_argument("--bare", default="lztmre_aus_m198912202012_dixa2_bareStats.tif", help="Bare stats raster")
    parser.add_argument("--ibra", default="ibra.fgb", help="IBRA vector dataset")
    parser.add_argument("-o", "--output", default="lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif", help="Output GeoTIFF")
    parser.add_argument("--test-chip", nargs=4, type=int, metavar=("XOFF", "YOFF", "W", "H"), help="Test chip window")
    return parser.parse_args()


def main():
    args = parse_args()

    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherArgs = applier.OtherInputs()

    infiles.changeData = args.change
    infiles.terrainData = args.terrain
    infiles.bareStats = args.bare
    outfiles.dcrmMask = args.output

    # Inspect reference dataset
    ref_ds = gdal.Open(infiles.bareStats, gdal.GA_ReadOnly)
    target_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()

    # Reproject IBRA vector to Australian Albers if needed
    ibra_albers_path = ensure_ibra_albers(args.ibra, target_proj)
    vec_ds = ogr.Open(ibra_albers_path, 0)
    otherArgs.lyr = vec_ds.GetLayer(0)
    otherArgs.proj = target_proj
    otherArgs.extent = otherArgs.lyr.GetExtent()

    controls = applier.ApplierControls()
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setReferenceImage(infiles.bareStats)
    controls.setResampleMethod("near")
    controls.setStatsIgnore(0)
    controls.setOutputDriverName("GTIFF")
    controls.setCreationOptions(["COMPRESS=DEFLATE",
                                 "ZLEVEL=1",
                                 "BIGTIFF=YES",
                                 "TILED=YES",
                                 "INTERLEAVE=BAND",
                                 "NUM_THREADS=ALL_CPUS",
                                 "BLOCKXSIZE=512",
                                 "BLOCKYSIZE=512"])
    controls.setWindowXsize(512)
    controls.setWindowYsize(512)
    controls.progress = cuiprogress.CUIProgressBar()

    if args.test_chip:
        xoff_t, yoff_t, w_t, h_t = args.test_chip
        xmin = ref_gt[0] + xoff_t * ref_gt[1]
        ymax = ref_gt[3] + yoff_t * ref_gt[5]
        ymin = ymax + h_t * ref_gt[5]
        xmax = xmin + w_t * ref_gt[1]
        grid = applier.PixelGridDefn(
            xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax,
            xRes=ref_gt[1], yRes=abs(ref_gt[5]), projection=target_proj
        )
        controls.setReferencePixgrid(grid)

    print(f"Applying maskStatsIbra -> {args.output}...")
    applier.apply(maskStatsIbra, infiles, outfiles, otherArgs, controls=controls)

    # Set band descriptions
    out_ds = gdal.Open(args.output, gdal.GA_Update)
    b1 = out_ds.GetRasterBand(1)
    b1.SetDescription("Masked Reference Groundcover (1..101, 0=NoData)")
    b1.SetNoDataValue(0)
    b2 = out_ds.GetRasterBand(2)
    b2.SetDescription("IBRA Region Code (OBJECTID 1..89, 0=NoData)")
    b2.SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
    print(f"Successfully generated 2-band masked stats + IBRA raster: {args.output}")


if __name__ == "__main__":
    main()
