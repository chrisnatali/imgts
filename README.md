# Image Time Series Processing Library (imgts)

This library was written to streamline computing vegetation index statistics from image timeseries. The functionality here was meant to be comparable to that of the [extractVIs.R](https://github.com/gianlucafilippa/phenopix/blob/master/R/extractVIs.R) function from the [phenopix](https://github.com/gianlucafilippa/phenopix) R library. The metrics computations run orders of magnitude faster in the imgts library vs the corresponding R version.

## Usage

There are two CLI modules in this library:

### preprocess

This has commands for preparing data for `process`.

See `python -m imgts.preprocess --help` for usage details.

### process

For computing the greenness metrics over the image time series.

See `python -m imgts.process --help` for usage details.

### Example

This assumes you have a timeseries of images of the same site in a directory (referenced as `data/image_dir` below).

1. Use the phenopix library `DrawMULTIROI` function to generate ROIs (as described [here](https://github.com/jnatali/sagehen_meadows/blob/chrisnatali/poc_generate_vegetation_phenology_time_series/roi_vi_notes.md#drawmultiroi))

2. Extract the ROIs of interest from R output as json polygons

    - See example functions and script for this [here](https://github.com/jnatali/sagehen_meadows/blob/chrisnatali/poc_generate_vegetation_phenology_time_series/src/poly_helper.R#L4-L15)

    - Note that the json polygon coordinates will have their origin at the bottom left of the image

3. Create an ROI bit mask image from the json polygons

    - This represents the interiors of the polygons from step 2 as "true" values in the output image to act as a bit mask that can be applied to any image to extract the region of interest.

    - The `ref_image_path` should point to a representative image from your timeseries image directory to provide the bounds of the bit mask.

    - Run something like the following to get the bit mask in `data/out/roi-mask.png`:

```
python -m imgts.preprocess polygons_to_image --json_path data/roi-polys.json --ref_image_path data/image_dir/image1.jpg --output_path data/out/roi-mask.png
```
4. Create an image time series table

    - This will be used to drive the metric computation in the final step

    - Run something like the following to output the image timeseries table into `data/out/img_ts.csv`:

```
python -m imgts.preprocess image_dir_to_timeseries --image_dir data/image_dir --output_path data/out/img_ts.csv

```

5. Use the ROI bit mask and the image timeseries table to compute the vegetation metrics for the timeseries of images

    - Run something like the following to output the full image timeseries vegetation metrics into `data/out/img_ts_metrics.csv`:

```
python -m imgts.process --images data/img_ts.csv --mask data/roi-mask.png --out data/out/img_ts_metrics.csv

```

## Testing

Some basic test data in this repo can be used for the following basic tests of functionality:

### preprocess

```
# polygons_to_image
# NOTE: This assumes the polygons have coordinate origin at bottom-left as output by phenopix R lib
python -m imgts.preprocess polygons_to_image --json_path test/data/roi-polys-raw.json --ref_image_path test/data/image.jpg --output_path test/data/out/roi-mask.png

# apply_mask_and_write
python -m imgts.preprocess apply_mask_and_write --mask_path test/data/roi-mask.png --image_path test/data/image.jpg --output_path test/data/out/masked-image.jpg

# image_dir_to_timeseries
python -m imgts.preprocess image_dir_to_timeseries --image_dir test/data/t1_images --output_path test/data/out/t1_img_ts.csv

# Then compare the files in test/data/out to corresponding files in test/data
```

### process

```
# Run for single image
python -m imgts.process --images test/data/single_img_ts.csv --mask test/data/roi-mask.png --out test/data/out/single_img_out.csv

# Run for t1 image directory set
python -m imgts.process --images test/data/t1_img_ts.csv --mask test/data/roi-mask.png --out test/data/out/t1_img_ts_out.csv

```

