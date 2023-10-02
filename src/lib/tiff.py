from osgeo import gdal


def print_geotiff_info(file_path):
    # Open the file
    ds = gdal.Open(file_path)

    if ds is None:
        print("Failed to open file: " + file_path)
        return

    # Print raster dataset information
    print("Raster dataset information:")
    print("  - Number of bands: {}".format(ds.RasterCount))
    print("  - Size is {} x {}".format(ds.RasterXSize, ds.RasterYSize))

    # Get the geotransform
    gt = ds.GetGeoTransform()
    print("\nGeotransform:")
    print("  - Origin: ({}, {})".format(gt[0], gt[3]))
    print("  - Pixel Size: ({}, {})".format(gt[1], gt[5]))

    # Get projection
    proj = ds.GetProjection()
    print("\nProjection/Coordinate System:")
    print("  - {}".format(proj))

    # Get metadata
    metadata = ds.GetMetadata()
    print("\nMetadata:")
    for key, value in metadata.items():
        print("  - {} : {}".format(key, value))

    # Close dataset
    ds = None


if __name__ == "__main__":
    print_geotiff_info("/mnt/data/Videos/vacaville/my_project0000.tif")
    print_geotiff_info("/mnt/data/Videos/vacaville/my_project0001.tif")
