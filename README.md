# Image Signal Processing Pipeline

This project implements an Image Signal Processing (ISP) pipeline to process raw image data captured by Nikon D3500. It can be adpated for other cameras, but you will need to modify the camera specific parameters accordingly.  

The pipeline includes several stages such as:

Linearization  
Auto and Manual White Balance  
Demosaicing  
Color Correction  
Brightness Adjustment  
Gamma Correction  
Image Compression  

## Prerequisites

The following Python packages are required to run the code:  

numpy  
scipy  
matplotlib  
scikit-image

## How to Use

### Input Image:
The pipeline takes a raw image in TIFF format as input. 

### Linearization:
This step adjusts the pixel values based on the camera's darkness and saturation values.

### Automatic White Balance (AWB): 
Adjusts the red and blue channel intensities based on the gray or white world assumption, or by using pre-calculated dcraw parameters.

### Manual White Balance (MWB): 
Two points from the image can be selected, which are assumed to represent white. The Manual White Balance (MWB) algorithm then performs white balance on the entire image based on the color values of these selected points. 

### Demosaicing:
The raw Bayer pattern is interpolated into a full RGB image using bilinear interpolation.

### Color Correction:
The image is corrected using XYZ color space transformations to map the image colors from the camera to the sRGB color space.

### Brightness Adjustment:
Adjusts the overall brightness of the image to match a target mean brightness value.

### Gamma Correction:
Applies gamma correction to the image to adjust its non-linear luminance.

### Compression:
The processed image is saved in both PNG and JPEG formats. The JPEG format can be compressed with a specified quality value.

### The pipeline generates the following files:

image_RGGB.png: Demosaiced and processed image  
color_corrected_image.png: Image after color correction  
brightness_adjusted_image.jpg: Image after brightness adjustment  
gamma_image.png: Image after gamma correction  
output.jpg: Compressed JPEG image  
output.png: Uncompressed PNG image  
The compression ratio between the uncompressed and compressed images will be printed after the compression step  

## License
This project is open-source and free to use.

