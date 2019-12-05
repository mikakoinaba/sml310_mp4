import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import array
from torch.autograd import Variable
import torch

imgList = os.listdir(path='./data/cropped64') # list of image names
images = []

for image in imgList:
	currMatrix = array(mpimg.imread('./data/cropped64/' + image)[:, :, :3])
	# black - white is 000 to 255 255 255 so average
	currMatrix = (currMatrix[:,:,0] + currMatrix[:,:,1] + currMatrix[:,:,2]) / 3
	images.append(currMatrix.flatten())

images = array(images) # Part 1 and 2 -- list of grayscale images matrices (64 by 64)

# FOR REPORT display a few images in subplot
# print 4 Angies in a 2 by 2
def plot4Angies():
	subplotList = ['0', '53', '100', '151']
	numPics = 4
	for i in range(1, numPics + 1):
		plt.subplot(2, 2, i)
		plt.imshow(mpimg.imread('./data/cropped64/AngieHarmon' + subplotList[i - 1] + '.png')[:, :, :3])
		plt.axis('off')
	plt.show()

# FOR REPORT: yeah prob registered

# FOR REPORT example gray scale image
def plotExGrayscale():
	exImag = array(mpimg.imread('./data/cropped64/AngieHarmon0.png')[:, :, :3])
	exImag = (exImag[:,:,0] + exImag[:,:,1] + exImag[:,:,2]) / 3
	plt.imshow(exImag, cmap='gray')
	plt.axis('off')
	plt.show()

# Part 3
n = len(imgList)

idx = np.random.RandomState(seed=31).permutation(n)

tags = [0] * 152 # 'angie'
tags.extend([1] * 110) # 'daniel'
tags.extend([2] * 119) # 'gerard'
tags.extend([3] * 147) # 'lorraine'
tags.extend([4] * 134) # 'michael'
tags.extend([5] * 144) # 'peri'

tags = array(tags)

trainProp = 0.70
valProp = 0.15

trainNum = int(n * trainProp) # 564
valNum = trainNum + int(n * valProp) # 684

trainIdx = idx[:trainNum] # 0 - 563
validIdx = idx[trainNum:valNum] # 564 - 683
testIdx = idx[valNum:n] # 684 - 805

train_x = images[trainIdx]
train_y = tags[trainIdx]

valid_x = images[validIdx]
valid_y = tags[validIdx]

test_x = images[testIdx]
test_y = tags[testIdx]

dim_x = 64 * 64
dim_h = 30
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

x_train = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(train_y), requires_grad=False).type(dtype_long)

x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
y_validclasses = Variable(torch.from_numpy(valid_y), requires_grad=False).type(dtype_long)

x_test = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

# Part 3 & 4
def noHidden():
	model_logreg = torch.nn.Sequential(
		torch.nn.Linear(dim_x, dim_out)
	)

	loss_train = []
	loss_valid = [] 

	learning_rate = 1e-3
	
	N = 10000 
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)

	for t in range(N):
	    y_pred = model_logreg(x_train)
	    loss = loss_fn(y_pred, y_classes)
	    loss_train.append(loss.data.numpy().reshape((1,))[0]/len(y_classes))

	    y_validpred = model_logreg(x_valid)
	    loss_v = loss_fn(y_validpred, y_validclasses)
	    loss_valid.append(loss_v.data.numpy().reshape((1,))[0]/len(y_validclasses))

	    model_logreg.zero_grad() 
	    loss.backward()   
	    optimizer.step()   

	# FOR REPORT learning curve -- uhh looks ok? a little funky at the end
	plt.plot(range(N), loss_valid, color='b', label='validation')
	plt.plot(range(N), loss_train, color='r', label='training')
	plt.legend(loc='upper left')
	plt.title("Learning curve for training and validation sets")
	plt.xlabel("Number of iterations (N)")
	plt.ylabel("Cross entropy loss")
	plt.show()

	y_testpred = model_logreg(x_test).data.numpy()

	# FOR REPORT accuracy on test set -- 0.1885245901639344 -- this ok ? rip 
	print('accuracy on testing set: ', np.mean(np.argmax(y_testpred, 1) == test_y))

	# # FOR REPORT? weights from first 10 inputs to each actor output
	# print(model_logreg[0].weight.data.numpy()[0, :10]) # 'angie'
	# print(model_logreg[0].weight.data.numpy()[1, :10]) # 'daniel'
	# print(model_logreg[0].weight.data.numpy()[2, :10]) # 'gerard'
	# print(model_logreg[0].weight.data.numpy()[3, :10]) # 'lorraine'
	# print(model_logreg[0].weight.data.numpy()[4, :10]) # 'michael'
	# print(model_logreg[0].weight.data.numpy()[5, :10]) # 'peri'

	# FOR REPORT plot of weights
	nameList= ['Angie Harmon', 'Daniel Radcliffe', 'Gerard Butler', 'Lorraine Bracco', 'Michael Vartan', 'Peri Gilpin']
	for i in range(6):
		plt.subplot(2, 3, i + 1)
		plt.imshow(model_logreg[0].weight.data.numpy()[i, :].reshape((64, 64)), cmap='coolwarm')
		plt.axis('off')
		plt.title(nameList[i])
#	plt.suptitle('Weights per Actor')
	plt.show()

# Part 5 6 & 7 
def hidden():
	model = torch.nn.Sequential(
		torch.nn.Linear(dim_x, dim_h),
		torch.nn.ReLU(),
		torch.nn.Linear(dim_h, dim_out),
		)

	loss_train = []
	loss_valid = [] 

	learning_rate = 1e-3

	N = 10000 
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for t in range(N):
	    y_pred = model(x_train)
	    loss = loss_fn(y_pred, y_classes)
	    loss_train.append(loss.data.numpy().reshape((1,))[0]/len(y_classes))

	    y_validpred = model(x_valid)
	    loss_v = loss_fn(y_validpred, y_validclasses)
	    loss_valid.append(loss_v.data.numpy().reshape((1,))[0]/len(y_validclasses))

	    model.zero_grad() 
	    loss.backward()   
	    optimizer.step()   

	# FOR REPORT learning curve -- looks ok ? 
	plt.plot(range(N), loss_valid, color='b', label='validation')
	plt.plot(range(N), loss_train, color='r', label='training')
	plt.legend(loc='upper left')
	plt.title("Learning curve for training and validation sets")
	plt.xlabel("Number of iterations (N)")
	plt.ylabel("Cross entropy loss")
	plt.show()

	y_testpred = model(x_test).data.numpy()

	# FOR REPORT accuracy on test set -- 0.20491803278688525 -- rip idk
	print('accuracy on testing set: ', np.mean(np.argmax(y_testpred, 1) == test_y))

	# FOR REPORT plot of weights
	plt.rcParams.update({'font.size': 5})
	for i in range(30):
		plt.subplot(5, 6, i + 1)
		plt.imshow(model[0].weight.data.numpy()[i, :].reshape((64, 64)), cmap='coolwarm')
		plt.axis('off')
		plt.title('Unit ' + str((i + 1)))
#	plt.suptitle('Weights from Inputs to Hidden Units')
	plt.tight_layout(pad=0.1)
	plt.show()

noHidden()
# Part 7 -- he explained it but I wasn't paying attention rip
