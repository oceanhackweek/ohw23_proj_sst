import s3fs
import xarray as xr

# Bypass AWS tokens, keys etc.
s3 = s3fs.S3FileSystem(anon=True)

# Verify that we're in the right place
sst_files = s3.ls("mur-sst/zarr-v1/")
sst_files

ds = xr.open_zarr(
        store=s3fs.S3Map(
            root=f"s3://{sst_files[0]}", s3=s3, check=False
        )
)
