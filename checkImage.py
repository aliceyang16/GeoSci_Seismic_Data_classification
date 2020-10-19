import cv2

size = 128, 128

image_name = 'Figure_1.png_05_05_1.png'
img = cv2.imread(image_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

resized = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
cv2.imwrite('cropped_'+image_name, resized)