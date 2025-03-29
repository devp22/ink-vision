import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pad
import matplotlib.pyplot as plt

'''Neural Network Class'''
class CNN(nn.Module):
    def __init__(self,input_size=1,output_size=10): 
        super(CNN,self).__init__()
        # self.layer1 = nn.Linear(input_size,50)
        # self.layer2 = nn.Linear(50,50)
        # self.layer3 = nn.Linear(50,output_size)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Added one dense layer for better learning
        self.fc2 = nn.Linear(128, output_size)

    
    def forward(self,input_tensor): #In PyTorch, when defining a neural network using nn.Module, the standard method name for the forward pass is forward(). This is because PyTorch internally calls forward() when you pass data through the model. If you name the method anything other than forward(), PyTorch won't automatically call it
        # output_tensor = F.relu(self.layer1(input_tensor)) # put input tensor in layer 1 and then apply relu activation function
        # output_tensor = F.relu(self.layer2(output_tensor))
        # output_tensor = self.layer3(output_tensor)
        # return output_tensor

        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = self.dropout(output_tensor)
        output_tensor = output_tensor.view(output_tensor.size(0), -1)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor

image = Image.open('test_images/test_image3.png').convert("L")
image = ImageOps.invert(image)
image = image.resize((28, 28))
# image = pad(image, padding=4, fill=0, padding_mode='constant')  # total = 28x28

model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

transFormer = transforms.Compose([
        transforms.RandomRotation(10),  # Augmentation: slight rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Augmentation: random shifts
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization
    ])

image_tensor = transFormer(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    predicted_digit = torch.argmax(output,dim=1)
    print("digit is: "+str(predicted_digit.item()))

plt.imshow(image_tensor.view(28, 28), cmap='gray')
plt.show()
