import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2gray
import os

class ISP:
    def __init__(self, path):
        self.raw = io.imread(path)
        print(f'The image is of size {self.raw.shape[0]} x {self.raw.shape[1]} and has {self.raw.dtype.itemsize * 8} bits per pixel')
        self.raw = self.raw.astype('double')

    def linearize(self, darkness, saturation):
        self.raw = (self.raw - darkness) / (saturation - darkness)
        self.raw = np.clip(self.raw, 0, 1)
    
    def bayer_pattern(self):
        print("Bayer pattern of the image is RGGB. It is obtained by performing white balance and demosaicing with all different bayer patterns. Code for this is available in awb and demosaic")

    def awb(self, type, pattern):

        if pattern == 'RGGB':
            r = self.raw[::2, ::2]
            g1 = self.raw[::2, 1::2]
            g2 = self.raw[1::2, ::2]
            b = self.raw[1::2, 1::2]
        elif pattern == 'GRBG':
            g1 = self.raw[::2, ::2]
            r = self.raw[::2, 1::2]
            b = self.raw[1::2, ::2]
            g2 = self.raw[1::2, 1::2]
        elif pattern == 'GBRG':
            g1 = self.raw[::2, ::2]
            b = self.raw[::2, 1::2]
            r = self.raw[1::2, ::2]
            g2 = self.raw[1::2, 1::2]
        elif pattern == 'BGGR':
            b = self.raw[::2, ::2]
            g1 = self.raw[::2, 1::2]
            g2 = self.raw[1::2, ::2]
            r = self.raw[1::2, 1::2]

        g = np.concatenate((g1,g2), axis = 0)
        if(type == "gray"):
            r_scale = g.mean() / r.mean()
            b_scale = g.mean() / b.mean()
        elif (type == "white"):
            r_scale = g.max()/ r.max()
            b_scale = g.max() / b.max()
        elif(type == "dcraw"):
            r_scale = 2.394531
            b_scale = 1.597656

        self.raw[::2,::2] *= r_scale
        self.raw[1::2,1::2] *= b_scale 

    def demosaic(self, pattern):

        if pattern == 'RGGB':
            r = self.raw[::2, ::2]
            g1 = self.raw[::2, 1::2]
            g2 = self.raw[1::2, ::2]
            b = self.raw[1::2, 1::2]
        elif pattern == 'GRBG':
            g1 = self.raw[::2, ::2]
            r = self.raw[::2, 1::2]
            b = self.raw[1::2, ::2]
            g2 = self.raw[1::2, 1::2]
        elif pattern == 'GBRG':
            g1 = self.raw[::2, ::2]
            b = self.raw[::2, 1::2]
            r = self.raw[1::2, ::2]
            g2 = self.raw[1::2, 1::2]
        elif pattern == 'BGGR':
            b = self.raw[::2, ::2]
            g1 = self.raw[::2, 1::2]
            g2 = self.raw[1::2, ::2]
            r = self.raw[1::2, 1::2]

        x_r = np.linspace(0, self.raw.shape[1] - 2, r.shape[1])
        y_r = np.linspace(0, self.raw.shape[0] - 2, r.shape[0])
        x_g1 = np.linspace(0, self.raw.shape[1] - 2, g1.shape[1])
        y_g1 = np.linspace(1, self.raw.shape[0], g1.shape[0])
        x_g2 = np.linspace(1, self.raw.shape[1], g2.shape[1])
        y_g2 = np.linspace(0, self.raw.shape[0] - 2, g2.shape[0])
        x_b = np.linspace(1, self.raw.shape[1], b.shape[1])
        y_b = np.linspace(1, self.raw.shape[0], b.shape[0])
        
        red_interp = interp2d(x_r, y_r, r, kind='linear')
        g1_interp = interp2d(x_g1, y_g1, g1, kind='linear')
        g2_interp = interp2d(x_g2, y_g2, g2, kind='linear')
        blue_interp = interp2d(x_b, y_b, b, kind='linear')

        x_full = np.arange(0, self.raw.shape[1])
        y_full = np.arange(0, self.raw.shape[0])

        red_full = red_interp(x_full, y_full)
        g1_full = g1_interp(x_full, y_full)
        g2_full = g2_interp(x_full, y_full)
        g_full = (g1_full + g2_full) / 2
        b_full = blue_interp(x_full, y_full)

        full_image = np.stack((red_full, g_full, b_full), axis = -1)
        self.raw = full_image
        jpg_image = np.clip(full_image * 255 * 5, 0, 255).astype(np.uint8)
        io.imsave(f'image_{pattern}.png', jpg_image)

    def color_correction(self, xyz_to_cam, srgb_to_xyz):
        srgb_to_cam = xyz_to_cam @ srgb_to_xyz
        srgb_to_cam_norm = srgb_to_cam / (np.sum(srgb_to_cam, axis = 1, keepdims=True))
        cam_to_srgb = np.linalg.inv(srgb_to_cam_norm)
        image_height, image_width, _ = self.raw.shape
        self.raw_reshaped = np.reshape(self.raw, (-1, 3))
        color_corrected_image = (cam_to_srgb @ self.raw_reshaped.T).T
        self.raw = np.reshape(color_corrected_image, (image_height, image_width, 3))
        jpg_image = np.clip(self.raw * 255 * 5, 0, 255).astype(np.uint8)
        io.imsave("color_corrected_image.png", jpg_image)

    def brightness_adjust(self, target_mean):
        current_mean = np.mean(rgb2gray(self.raw))
        self.raw = np.clip((target_mean / current_mean) * self.raw, 0, 1)
        jpg_image = (self.raw * 255).astype(np.uint8)
        io.imsave("brightness_adjusted_image.jpg", jpg_image)

    def gamma(self):
        self.raw_gamma = np.where(self.raw <= 0.0031308, 12.92 * self.raw, (1 + 0.055) * (self.raw**(1 / 2.4)) - 0.055)
        self.raw = np.clip(self.raw_gamma, 0, 1)
        jpg_image = (self.raw * 255).astype(np.uint8)
        io.imsave("gamma_image.png", jpg_image)

    def compress(self, jpeg_quality, path):
        output_raw = (self.raw * 255).astype('uint8')
        io.imsave("output.png", output_raw)
        io.imsave("output.jpg", output_raw, quality = jpeg_quality)
        directory = os.path.dirname(path)
        uncompressed_img_path = os.path.join(directory, "output.png")
        compressed_img_path = os.path.join(directory, "output.jpg")

        uncompressed_img_size = os.path.getsize(uncompressed_img_path)
        compressed_img_size = os.path.getsize(compressed_img_path)
        print(uncompressed_img_size)
        print(compressed_img_size)
        print(f'compression ratio = {uncompressed_img_size / compressed_img_size}')

    def patch_bayer_pattern(self, xmin, ymin):
        if ymin % 2 == 0:
            if xmin % 2 == 0:
                top_left = 'R'
                top_right = 'G'
                bottom_left = 'G'
                bottom_right = 'B'
            else:
                top_left ='G'
                top_right = 'R'
                bottom_left = 'B'
                bottom_right = 'G'
        else:
            if xmin % 2 == 0:
                top_left = 'G'
                top_right = 'B'
                bottom_left = 'R'
                bottom_right = 'G'
            else:
                top_left ='B'
                top_right = 'G'
                bottom_left = 'G'
                bottom_right = 'R'
        return top_left, top_right, bottom_left, bottom_right

    def manual_white_balance(self):
        plt.imshow(self.raw, cmap='gray')
        plt.title("select a patch by clicking on two points in the image")
        patch = plt.ginput(2)
        plt.close()
        x1, y1 = int(patch[0][0]), int(patch[0][1])
        x2, y2 = int(patch[1][0]), int(patch[1][1])
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        patch = self.raw[y1:y2, x1:x2]
        top_left, top_right, bottom_left, bottom_right = self.patch_bayer_pattern(xmin, ymin)
        if top_left == 'R':
            r_patch = patch[::2, ::2]
            g1_patch = patch[::2, 1::2]
            g2_patch = patch[1::2, ::2]
            b_patch = patch[1::2, 1::2]
        elif top_left == 'G':
            if top_right == 'R':
                g1_patch = patch[::2, ::2]
                r_patch = patch[::2, 1::2]
                b_patch = patch[1::2, ::2]
                g2_patch = patch[1::2, 1::2]
            else:
                g1_patch = patch[::2, ::2]
                b_patch = patch[::2, 1::2]
                r_patch = patch[1::2, ::2]
                g2_patch = patch[1::2, 1::2]
        elif top_left == 'B':
            b_patch = patch[::2, ::2]
            g1_patch = patch[::2, 1::2]
            g2_patch = patch[1::2, ::2]
            r_patch = patch[1::2, 1::2]

        min_height = min(g1_patch.shape[0], g2_patch.shape[0])
        min_width = min(g1_patch.shape[1], g2_patch.shape[1])
        g1_patch = g1_patch[:min_height, :min_width]
        g2_patch = g2_patch[:min_height, :min_width]
        g_patch = (g1_patch + g2_patch) / 2
        r_scale = g_patch.mean() / r_patch.mean()
        b_scale = g_patch.mean() / b_patch.mean()

        self.raw[::2,::2] *= r_scale
        self.raw[1::2,1::2] *= b_scale 



if __name__ == "__main__":
    image_path = "/Users/revanthvarma/Desktop/15663_CP/assgn1/data/campus.tiff"
    imagePipeline = ISP(image_path)
    imagePipeline.linearize(150, 4095)
    imagePipeline.bayer_pattern()
    imagePipeline.awb("dcraw", pattern = 'RGGB')
    # imagePipeline.manual_white_balance()
    imagePipeline.demosaic(pattern = 'RGGB')
    xyz_to_cam = 1e-4 * np.reshape([6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318], (3,3))
    srgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    imagePipeline.color_correction(xyz_to_cam, srgb_to_xyz)
    imagePipeline.brightness_adjust(0.15)
    imagePipeline.gamma()
    imagePipeline.compress(45, image_path)


