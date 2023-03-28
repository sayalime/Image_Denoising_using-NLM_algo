import cv2
from denoise_img import non_local_means_denoise

input_image = cv2.imread('girl.jpg')


h = 0.2
sigma = 0.5


output_image = non_local_means_denoise(input_image, h=h, sigma=sigma)


cv2.imwrite('output_image.jpg', output_image)
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
