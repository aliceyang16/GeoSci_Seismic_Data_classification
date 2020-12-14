import numpy as np
import matplotlib.pyplot as plt
import os

directory = os.getcwd()
pngDir = './png/'

def plot2d(gx,fx,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  #ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

def saveImage(image_name, image):
  plt.axis('off')
  plt.imshow(image)
  plt.savefig('./images/seis/'+image_name+'.png')


#datatype = 'validation'
datatype = 'train'

seismPath = './data/'+ datatype+'/seis/'
faultPath = './data/'+ datatype+'/fault/'
n1,n2,n3=128,128,128
dk = 0
gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
gx = np.reshape(gx,(n1,n2,n3))
fx = np.reshape(fx,(n1,n2,n3))
gm = np.mean(gx)
gs = np.std(gx)
gx = gx-gm
gx = gx/gs

gx = np.transpose(gx)
fx = np.transpose(fx)
#fp = model.predict(np.reshape(gx,(1,n1,n2,n3,1)),verbose=1)
#fp = fp[0,:,:,:,0]

for i in range(128):
	# Generates seismic images
  imageName1 = 'top_view' + str(i)
  gx1 = gx[i, :, :]
  saveImage(imageName1, gx1)
  imageName2 = 'width_view' + str(i)
  gx2 = gx[:, i, :]
  saveImage(imageName2, gx2)
  imageName3 = "height_view" + str(i)
  gx3 = gx[:, :, i]
  saveImage(imageName3, gx3)

  # Generates highlighted faults
  # imageName1 = 'top_view' + str(i)
  # fx1 = fx[i, :, :]
  # saveImage(imageName1, fx1)
  # imageName2 = 'width_view' + str(i)
  # fx2 = fx[:, i, :]
  # saveImage(imageName2, fx2)
  # imageName3 = "height_view" + str(i)
  # fx3 = fx[:, :, i]
  # saveImage(imageName3, fx3)


#gx1 = gx[50,:,:]
#fp1 = fp[50,:,:]
#gx2 = gx[:,29,:]
#fp2 = fp[:,29,:]
#gx3 = gx[:,:,29]
#fp3 = fp[:,:,29]
#plot2d(gx1,fx1,png='fp1')
#plot2d(gx2,fx2,png='fp2')
#plot2d(gx3,fx3,png='fp3')