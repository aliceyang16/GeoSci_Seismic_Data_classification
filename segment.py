# Image Segmentation in Python
import image_slicer
import os
import numpy
import cv2
import shutil
from PIL import Image 

# def sliceImages():
# 	print ("Slicing Images ... ")
# 	rootdir = os.getcwd()+"\\cropped_images\\"
# 	for subdir, dirs, files in os.walk(rootdir):
# 		for image in files:
# 			onlyfiles=''
# 			filename, file_extension = os.path.splitext(image)
# 			if file_extension == ".jpg":
# 				onlyfiles=str(onlyfiles)+str(image)

# 			if onlyfiles!='':
# 				tileImageFolderName = rootdir+filename+"\\"
# 				print (filename+"\n")
# 				if not os.path.exists(tileImageFolderName):
# 					os.makedirs(tileImageFolderName)
# 					os.makedirs(tileImageFolderName+"Leg\\")
# 					os.makedirs(tileImageFolderName+"Foot\\")
# 					os.makedirs(tileImageFolderName+"Fracture\\")
# 					os.makedirs(tileImageFolderName+"Knee\\")
# 					os.makedirs(tileImageFolderName+"Blank\\")
# 				image_tiles = image_slicer.slice(rootdir+onlyfiles,9, save=False)
# 				image_slicer.save_tiles(image_tiles, directory=tileImageFolderName, prefix=filename)
# 				shutil.move(rootdir+onlyfiles, tileImageFolderName+onlyfiles)
# 		break
# 	return 0

root = os.getcwd()
size = 128, 128

def grayscaleConversion(filename):
	img = Image.open(filename).convert('LA')
	img.save(filename)

def cropImage(filename):
	left = 82
	top = 60
	right = 573
	bottom = 421

	img = Image.open(filename)
	cropped_img = img.crop((left, top, right, bottom))
	cropped_img.save(filename)

def sliceImages(foldername, image_name):
	directory = root + "\\"+foldername+"\\"
	os.makedirs(directory)
	image_tiles = image_slicer.slice(image_name, 50, save=False)
	image_slicer.save_tiles(image_tiles, directory=directory, prefix=image_name)

def resizeImageCV(folderName):
	for filename in os.listdir(root+"\\"+folderName):
		if filename.endswith(".png"):
			print(filename)
			img = cv2.imread(root+"\\"+folderName+"\\"+filename)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
			cv2.imwrite(root+"\\"+folderName+"\\"+filename, resized)

if __name__ == '__main__':
	filename = 'Deep_Orange_edge detection_seismic_image_4.png'
	foldername = filename[:len(filename)-4]
	cropImage(filename)
	#grayscaleConversion(filename)
	sliceImages(foldername, filename)
	resizeImageCV(foldername)