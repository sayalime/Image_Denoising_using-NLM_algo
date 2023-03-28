import numpy as np
from scipy import ndimage
import cv2


def non_local_means_denoise(image, h=0.1, sigma=1.0):
    """
    Applies non-local means denoising to an input image.
    
    :param image: The input image to be denoised.
    :param h: The smoothing parameter. A larger value results in more smoothing.
    :param sigma: The standard deviation of the Gaussian noise in the image.
    :return: The denoised image.
    """
    
    image = np.float32(image) / 255.0

    
    window_size = int(np.ceil(3 * sigma))


    pad_width = ((window_size, window_size), (window_size, window_size), (0, 0))
    padded_image = np.pad(image, pad_width, mode='symmetric')
    kernel = np.exp(-0.5 * np.square(np.arange(-window_size, window_size+1, dtype=np.float32)) / np.square(sigma))
    kernel /= np.sum(kernel)

   
    output_image = np.zeros_like(image)

   
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
          
            search_window = padded_image[y:y+2*window_size+1, x:x+2*window_size+1, :]

        
            distance_squared = np.sum(np.square(search_window - padded_image[y+window_size, x+window_size]), axis=2)
            weight_matrix = np.exp(-distance_squared / (h**2))

         
            normalization_factor = np.sum(weight_matrix)
            search_window_reshaped = np.reshape(search_window, ((2*window_size+1)**2, 3))
            denoised_value = np.sum(weight_matrix.reshape(-1, 1) * search_window_reshaped, axis=0) / normalization_factor

      
            output_image[y, x, :] = denoised_value

   
    output_image = np.uint8(np.clip(output_image * 255.0, 0, 255))

    return output_image



