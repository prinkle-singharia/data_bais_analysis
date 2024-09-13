import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")

print(device)

# define CNN for a 3-class problem with input size 160x160 images
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(256 * 5 * 5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 3)
		self.relu = nn.ReLU()
		self.final_activation = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		x = self.pool(self.relu(self.conv5(x)))
		x = x.view(-1, 256 * 5 * 5)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.final_activation(x)
		return x



# Load dataset
train_dir = './data/train'
test_dir = './data/test'
image_size = 160
batch_size = 16
workers = 0

class CropToSmallerDimension(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        # Get the original image size
        width, height = img.size

        # Determine the smaller dimension
        smaller_dimension = min(width, height)

        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

print('Number of training images: {}'.format(len(train_dataset)))
print('Number of test images: {}'.format(len(test_dataset)))
print('Detected Classes are: ', train_dataset.classes) # classes are detected by folder structure

#Author: Prinkle Singharia, Discussed with: Pranav Nair and Christopher Dominic.

# Define the attack
def FGSM(model, image, label, epsilon):
	model.eval()
	
	image = Variable(image, requires_grad=True)
	
	# Forward pass
	output = model(image)
	
	loss = F.nll_loss(output, label)
	
	# Backward pass
	model.zero_grad()
	
	loss.backward()
	
	perturbed_image = image + (epsilon * image.grad.data.sign())
	
	# Clipping
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	
	return perturbed_image

def PGD(model, image, label, epsilon, iterations, alpha):
	model.eval()
	
	image = Variable(image, requires_grad=True)
	
	for _ in range(iterations):
		output = model(image)
		loss = F.nll_loss(output, label)
		# Backward pass
		model.zero_grad()
		loss.backward()
		data_grad = image.grad.data

	        perturbed_image = image + alpha * data_grad.sign()
	        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
	        perturbed_image = torch.clamp(perturbed_image, 0, 1)
	
	        # Update the image variable
	        image = Variable(perturbed_image, requires_grad=True)

    	return perturbed_image

net = Net()
net.to(device)

# Train the network

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# epochs = 100
# running_loss = 0
# train_losses, test_losses = [], []
# i=0

# for epoch in tqdm(range(epochs)):
# 	for inputs, labels in train_dataloader:
# 		inputs, labels = inputs.to(device), labels.to(device)
# 		optimizer.zero_grad()
# 		logps = net(inputs)
# 		loss = criterion(logps, labels)
# 		loss.backward()
# 		optimizer.step()
# 		running_loss += loss.item()

# # Save the model
# torch.save(net.state_dict(), 'model.pth')


# Test the model
net.load_state_dict(torch.load('model.pth', map_location="cpu"))
net.to(device)

correct=[]

net.eval()
accuracy = 0
for inputs, labels in tqdm(test_dataloader):
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = net(inputs)
	_, predicted = torch.max(outputs.data, 1)
	accuracy += (predicted == labels).sum().item()
	correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / len(test_dataset)))


# Test the model with adversarial examples
# Save adversarial examples for each class using FGSM with (eps = 0.01, 0.05, 0.1)
# Save one adversarial example for each class using PGD with (eps = 0.01, 0.05, 0.1, alpha = 0.001, 0.005, 0.01 respectively, iterations = 20)

fgsm_eps = [0.001, 0.01, 0.1]
alpha_values = [0.001, 0.01, 0.1]

for epsilon in fgsm_eps:
    accuracy = 0

    for input_image, labels in tqdm(test_dataloader):
        input_image, labels = input_image.to(device), labels.to(device)

        perturbed_inputs = FGSM(net, input_image, labels, epsilon)

        output_image = net(perturbed_inputs)
        _, predicted = torch.max(output_image.data, 1)
        accuracy += (predicted == labels).sum().item()

    print(f'Accuracy on adversarial test images for (epsilon={epsilon}): {100 * accuracy / len(test_dataset)} %')

class_names = train_dataset.classes

# Function to display images
def imshow(img, title):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Function to generate and display adversarial examples
def generate_and_display_adversarial_examples_fgsm(model, data_loader, alpha_values):
    model.eval()

    for class_index in range(len(class_names)):
        original_image, original_label = None, None
        for inputs, labels in data_loader:
            if labels[0] == class_index:
                original_image, original_label = inputs[0], labels[0]
                break

        imshow(vutils.make_grid(original_image, nrow=4), f'Original: {class_names[original_label]}')

        for alpha in alpha_values:
            perturbed_inputs = FGSM(model, original_image.unsqueeze(0).to(device), torch.tensor([original_label]), alpha)
            outputs = model(perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            imshow(vutils.make_grid(perturbed_inputs.cpu().detach(), nrow=4),
                   f'FGSM Adversarial Image for (α={alpha}) Predicted class: {class_names[predicted.item()]}')


generate_and_display_adversarial_examples_fgsm(net, test_dataloader, alpha_values)

accuracy = 0
pgd_eps = [0.01, 0.05, 0.1]
pgd_alpha_value = 2 / 255
pgd_iterations = 50

for epsilon in pgd_eps:
	accuracy = 0
	for inputs, labels in tqdm(test_dataloader):
		inputs, labels = inputs.to(device), labels.to(device)
		perturbed_inputs_pgd = PGD(net, inputs, labels, epsilon, pgd_iterations, pgd_alpha_value)
		outputs_pgd = net(perturbed_inputs_pgd)
		_, predicted_pgd = torch.max(outputs_pgd.data, 1)
		accuracy += (predicted_pgd == labels).sum().item()
		
	print('Accuracy of the network on adversarial test images: %d %%' % (100 * accuracy / len(test_dataset)))


# To generate Adversarial Images using PGD.
def generate_and_display_pgd_adversarial_examples(model, data_loader, epsilon_values, alpha, iterations):
	model.eval()
	for class_index in range(len(class_names)):
		original_image, original_label = None, None
		for inputs, labels in data_loader:
			if labels[0] == class_index:
				original_image, original_label = inputs[0], labels[0]
				break

        	# Show image
        	imshow(vutils.make_grid(original_image, nrow=4), f'Original Image: {class_names[original_label]}')

        	# Generate and display PGD adversarial examples for different epsilon values
        	for epsilon in epsilon_values:
            		# Wrap original_label in a tensor with batch dimension
            		perturbed_inputs_pgd = PGD(model, original_image.unsqueeze(0).to(device), torch.tensor([original_label]), epsilon, iterations, alpha)
            		outputs_pgd = model(perturbed_inputs_pgd)
            		_, predicted_pgd = torch.max(outputs_pgd.data, 1)

            		# Show adversarial image
            		imshow(vutils.make_grid(perturbed_inputs_pgd.cpu().detach(), nrow=4), f'PGD Adversarial Image for (ε={epsilon}) Predicted class: {class_names[predicted_pgd.item()]}')

# Call above method
epsilon_values_pgd = [0.01, 0.05, 0.1]
iterations_pgd = 20
generate_and_display_pgd_adversarial_examples(net, test_dataloader, epsilon_values_pgd, alpha_pgd, iterations_pgd)


adversarial_epochs = 10
epsilon_pgd_train = 0.075
alpha_pgd_train = 2 / 255
iterations_pgd_train = 50

for epoch in tqdm(range(adversarial_epochs)):
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        perturbed_inputs_pgd_train = PGD(net, inputs, labels, epsilon_pgd_train, iterations_pgd_train, alpha_pgd_train)

        logps = net(perturbed_inputs_pgd_train)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the updated model weights
torch.save(net.state_dict(), 'adversarial_model.pth')

# Load the above saved model
net.load_state_dict(torch.load('adversarial_model.pth', map_location="cpu"))
net.to(device)

correct=[]

net.eval()
accuracy = 0
for inputs, labels in tqdm(test_dataloader):
    print(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy += (predicted == labels).sum().item()
    correct.append((predicted == labels).tolist())

print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / len(test_dataset)))




























