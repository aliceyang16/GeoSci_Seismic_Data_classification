from PIL import Image 
import os
import cv2

def cropImage(folderName):
	left = 143
	top = 58
	right = 513
	bottom = 427

	name = './images/'+folderName+"/cropped/"
	if not os.path.exists(name):
		os.makedirs(name)

	directory = './images/' + folderName
	for filename in os.listdir(directory):
		if filename.endswith('.png'):
			print(filename)
			img = Image.open('./images/'+folderName+'/'+filename)
			cropped_img = img.crop((left, top, right, bottom))
			cropped_img.save('./images/' + folderName + '/cropped/' + filename)


def resizeImageCV(folderName, method, size):
	directory = './images/'+folderName+"\\resized\\"
	if not os.path.exists(directory):
		os.makedirs(directory)

	for filename in os.listdir('./images/'+folderName+"\\cropped\\"):
		if filename.endswith(".png"):
			print(filename)
			img = cv2.imread('./images/'+folderName+"\\cropped\\"+filename)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(img, size, interpolation = method)
			resized_filename = directory+"\\"+filename
			cv2.imwrite(resized_filename, resized)

def grayscaleConversion(folderName):
	directory = './images/'+folderName+"/grayscale/"
	if not os.path.exists(directory):
		os.makedirs(directory)

	for filename in os.listdir('./images/'+folderName+'/resized'):
		if filename.endswith('.png'):
			print(filename)
			img = Image.open('./images/'+folderName+'/resized/'+filename).convert('LA')
			img.save(directory+filename)


def createFolders(folderName):
	directory = folderName+"\\resized\\"
	if not os.path.exists(directory):
		os.makedirs(directory)

if __name__ == '__main__':
	size = 128, 128
	filename = 'seis'
	cropImage(filename)
	resizeImageCV(filename, cv2.INTER_NEAREST, size)
	grayscaleConversion(filename)


# pathName = os.getcwd()
# folderList = ["l-3", "l-2", "l-1", "l0", "l1", "l2", "l3"]

# folderMethodName = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
# resizeMethod = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

# size = 64, 64
# actions = ["training", "testing"]

# for folder in folderList:
# 	for action in actions:
# 		folderName = pathName+"\\"+folder+"\\"+action
# 		createFolders(folderName)
# 		resizeImageCV(folderName, folderMethodName[0], resizeMethod[0], size)