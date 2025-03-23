import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch
from PIL import Image

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

# trainset = datasets.MNIST(root='./data',train=True,download=True)

if __name__ == "__main__":  # to fix multiprocessing issue

    '''Creating transformer to normalize data'''
    # The MNIST dataset has pixel values in the range [0, 255]. Normalization scales them to [0, 1] (via ToTensor()) and further to zero-centered values using (mean=0.1307, std=0.3081), which improves model training.

    # print(f"trainset mean: {trainset.data.float().mean()/255}") # 0.1306
    # print(f"trainset std: {trainset.data.float().std()/255}") # 0.308

    transFormer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)) # Normalization is done with mean (0.1307) and std (0.308)
        ])
    print("Transformer created")
    '''Loading the data set'''
    # Dataset is loaded from MNIST library and is already splitted in training and testinf

    train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transFormer)
    test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transFormer)
    print("Datasets created")

    # print(f"dataset has been loaded")
    # print(f"length of training dataset is {len(train_dataset)}")
    # print(f"length of testing dataset is {len(test_dataset)}")

    ''' Creating DataLoaders to efficiently batch and shuffle '''
    # The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
    # You need DataLoader to create mini-batches and shuffle the data for training efficiency.

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True,num_workers=2,drop_last=True)
    print("DataLoaders created")

    ''''Hardware Availability'''
    # torch.cuda.is_available() → Checks if a GPU with CUDA support is available.
    # If a GPU is available, it assigns device = 'cuda', meaning the model and tensors will be processed on the GPU.
    # If no GPU is available, it assigns device = 'cpu', meaning computations will run on the CPU.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device} selected")

    ''''Model Creation'''

    # mnist images are 28x28 so input size is 784, and output classes are 10 whihcc represents one from each digit 0 to 9 for mnist
    model = CNN().to(device=device) 

    # A loss function is used to measure how well the CNN is performing on the training data. The loss function is typically calculated by taking the difference between the predicted labels and the actual labels of the training images.
    # An optimizer is used to update the weights of the CNN in order to minimize the loss function.
    # Learning Rate is the step size used for each iteration of the model training process. A learning rate that's too high can cause the model to converge too quickly to a suboptimal solution, and one that's too low might make the training process too slow.
    # num_epochs indicates the number of times the entire dataset is passed forward and backward through the neural network.
    # optim initializes an Adam optimizer, a popular choice for training deep learning models. Adam combines the advantages of two other extensions of stochastic gradient descent: Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).

    loss_fuction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001) # lr means learning rate

    print("Model Created")

    '''Training the Model'''

    for epoch in range(5): # Each epoch represents one complete pass over the entire training dataset. We will choose 5 i.e. 5 times whole dataset will be loaded
        print(f"Epoch no.: {epoch}")
        for batch, (data,targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # flattening not needed for CNN
            # data = data.reshape(data.shape[0],-1) # image is flattened for training

            # move forward
            scores = model(data)
            loss = loss_fuction(scores,targets)

            # move backward and tune gradients to zero for each batch i.e. backpropagation

            optimizer.zero_grad() # Before the backward pass, all the gradients of the model’s parameters are reset to zero. This is necessary because, by default, gradients are accumulated in PyTorch
            loss.backward() #  computes the gradient of the loss with respect to all trainable parameters in the model. These gradients are used by the optimizer to update model parameters.

            # grafient descent (The optimizer updates the model parameters using the gradients computed during backpropagation according to the Adam optimization algorithm and the specified learning rate.)
            optimizer.step()

    print("Training complete")

    '''Model Evaluation'''

    def check_accuracy(loader,model):

        model.eval()
        no_of_corrects = 0
        no_of_samples = 0
        with torch.no_grad(): # do not compute gradients
            for x,y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                # no flattening needed for CNN
                # x = x.reshape(x.shape[0],-1)

                scores = model(x)
                _,predictions = scores.max(1)

                no_of_corrects += (predictions==y).sum()
                no_of_samples += predictions.size(0)
            print(f'Got {no_of_corrects}/{no_of_samples} with accuracy {float(no_of_corrects)/float(no_of_samples)*100: .2f}')
        model.train()

    print("Accuracy for training dataset:")
    check_accuracy(train_loader,model)
    print("Accuracy for test dataset:")
    check_accuracy(test_loader,model)

    torch.save(model.state_dict(),'model.pth')
    print("model saved as model.pth")

