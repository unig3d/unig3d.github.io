# Background

Due to the large storage space required by the entire dataset, we have not found a suitable way to release the quadruples data other than text to the public. However, we provide a detailed data transformation pipeline. Based on this pipeline, users can easily obtain data consistent with our dataset.

Please configure the Blender environment on your machine first.

# 0-1.Mesh

Below is a script that can be run within Blender to render a 3D model as an 'obj' format mesh. Example usage: `blender -b -P glb2obj.py input_path`

**@liuzexiang**

# 2.Image

Script to run within Blender to render a 3D model as RGBAD images. Example usage:  `blender -b -P glb2rgbd.py input_path output_dir`

Pass `--camera_pose z-circular-elevated` for the rendering used to rendering the images with z-circular camera pose. Pass `--camera_pose random` for the rendering used to rendering the images with random camera pose. 

The output directory will include metadata json files for each rendered view, as well as a global metadata file for the render. Each image will be saved as a collection of 16-bit PNG files for each channel (RGBAD), as well as a full grayscale render of the view.

If the rendered images are found to be relatively dark, please use the 'img_bright.py' script to increase the image brightness. Example usage: `python img_bright.py input_dir`. The input directory refers to the `output_dir` of previous step.

# 3.Point Cloud

To convert the 3D models into colored point clouds, we utilize the RGBAD images with random camera poses obtained in the second step.

Example usage:  `python rgbd2pcd.py output_dir`

`output_dir` refers to the output directory of the second step.

# 4.Text

**@sunqinghong**
