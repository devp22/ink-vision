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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Same convolutions -> output -> same as input dimension
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # last conv out was 8 so this conv input is 8.

        self.fc1 = nn.Linear(16*7*7, output_size)
        # in fc1 -> 16 bcoz that is outchannel in conv2 and 7*7 because -> 2 poolings will make it (28/2)/2 => 7

    
    def forward(self,input_tensor): #In PyTorch, when defining a neural network using nn.Module, the standard method name for the forward pass is forward(). This is because PyTorch internally calls forward() when you pass data through the model. If you name the method anything other than forward(), PyTorch won't automatically call it
        # output_tensor = F.relu(self.layer1(input_tensor)) # put input tensor in layer 1 and then apply relu activation function
        # output_tensor = F.relu(self.layer2(output_tensor))
        # output_tensor = self.layer3(output_tensor)
        # return output_tensor

        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
        output_tensor = self.fc1(output_tensor)

        return output_tensor

image = Image.open('test_images/test_image3.png').convert("L")
image = ImageOps.invert(image)
image = image.resize((28, 28))
# image = pad(image, padding=4, fill=0, padding_mode='constant')  # total = 28x28

model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

transFormer = transforms.Compose([
        # transforms.Grayscale(),
        # transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)) # Normalization is done with mean (0.1307) and std (0.308)
        ])

image_tensor = transFormer(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    predicted_digit = torch.argmax(output,dim=1)
    print("digit is: "+str(predicted_digit.item()))

plt.imshow(image_tensor.view(28, 28), cmap='gray')
plt.show()
