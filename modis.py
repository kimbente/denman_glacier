import torch
import rasterio
import numpy as np
import pandas as pd

tif_path_image_hr = "/home/kim/data/modis/moa125_2014_hp1_v01.tif"

# The hr image takes a while to load... (7 minutes)
with rasterio.open(tif_path_image_hr) as src:
    # Read the data (for single-band TIFs, read the first band)
    image_hr = src.read(1)

image_hr_tensor = torch.tensor(image_hr.astype(np.float16), dtype = torch.int16)
del image_hr # Free memory

# Extract dims
n_rows = image_hr_tensor.shape[0]
n_columns = image_hr_tensor.shape[1]

print(image_hr_tensor.shape)
print()

x_min = -3174450
x_max = x_min + n_columns * 125
x = torch.linspace(start = x_min, end = x_max, steps = n_columns)
print("X shape:", x.shape)
print("Note, Antarctica is wider than it is tall.")

y_max = 2406325
y_min = y_max - n_rows * 125
# NOTE: need to go from max to min to index from top left
y = torch.linspace(start = y_max, end = y_min, steps = n_rows)
print("Y shape:", y.shape)

XX, YY = torch.meshgrid(x, y, indexing = 'xy')
XX.shape
# Order: X, Y, image

# Crashes sometimes
image_hr_grid = torch.concat((XX.unsqueeze(0), YY.unsqueeze(0), image_hr_tensor.unsqueeze(0)), dim = 0)

# function
def subset_tensor(tensor, x_min, x_max, y_min, y_max):
    """
    Subset a 3D tensor with shape (3, n_rows, n_columns) based on x and y ranges.
    """
    x_mask = (tensor[0, 0, :] >= x_min) & (tensor[0, 0, :] <= x_max)
    y_mask = (tensor[1, :, 0] >= y_min) & (tensor[1, :, 0] <= y_max)

    # Convert to indices
    x_inds = torch.where(x_mask)[0]
    y_inds = torch.where(y_mask)[0]

    # Use min/max to slice continuously
    x_min_idx, x_max_idx = x_inds[0], x_inds[-1] + 1
    y_min_idx, y_max_idx = y_inds[0], y_inds[-1] + 1

    # NOTE: WE return a clone
    return tensor[:, y_min_idx:y_max_idx, x_min_idx:x_max_idx].clone()

x_min = 2400000
x_max = 2700000
y_min = -550000
y_max = -250000

denman_subset = subset_tensor(
    image_hr_grid, 
    x_min, 
    x_max, 
    y_min, 
    y_max)

torch.save(denman_subset, '/home/kim/data/modis/moa125_2014_hp1_v01_denman_with_grid.pt')