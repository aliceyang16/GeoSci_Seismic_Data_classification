import numpy as np
import pandas as pd
import csv

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def writeToCSV(filename):
	fields = ['x', 'y', 'z']
	allData = []
	with open('dataXYZ.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(fields)
		with open(filename) as input:
			for l in input:
				line = l.split(' ')
				if line[0] is not '#':
					data = [float(line[0]), float(line[1]), float(line[2])]
					allData.append(data)
		input.close()
		csvwriter.writerows(allData)
	file.close()

def getAxes(filename):
	x = []
	y = []
	z = []
	with open(filename) as input:
		for l in input:
			line = l.split(' ')
			if line[0] is not '#':
				x.append(float(line[0]))
				y.append(float(line[1]))
				z.append(float(line[2]))
				
	return x, y, z

def getAxesCSV(filename):
	x = []
	y = []
	z = []
	with open(filename) as file:
		for row in file:
			line = row.split(',')
			x.append(float(line[0]))
			y.append(float(line[1]))
			z.append(float(line[2]))
	return x, y, z

def conditionalAxes(filename):
	x = []
	y = []
	z = []
	with open(filename) as input:
		for l in input:
			line = l.split(' ')
			if line[0] is not '#':
				if (float(line[4]) >= 6000) and (float(line[4]) <= 7500):
					x.append(float(line[0]))
					y.append(float(line[1]))
					z.append(float(line[2]))
				
	return x, y, z

def plot3DGraph(x, y, z):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(x, y, z, c=z, cmap='jet')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

def plot2DGraph(x, y, z):
	size = 682, 447
	plt.figure()
	plt.scatter(x, y, c=z, cmap='jet')
	#plt.xlabel('x')
	#plt.ylabel('y')
	#plt.colorbar()
	#plt.show()
	plt.savefig('seismic_image.png')

if __name__ == '__main__':
	filename = 'Deep_Orange_edge detection'
	#filename = 'Orebody-surface_EarthVision_Grid'
	#filename = 'VCR-OREBODY-Dip-Attribute-EarthVisionGrid(ASCII)'
	#x, y, z = getAxes(filename)

	# filename = 'dataXYZ.csv'
	x, y, z = conditionalAxes(filename)

	#plot3DGraph(x, y, z)
	plot2DGraph(x, y, z)
