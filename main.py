import xarray

ds = xarray.open_dataset("spinup.nc")
print(ds)