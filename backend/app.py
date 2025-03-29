from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import io

# Define the CNN model
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

# Set up Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Transform for MNIST-style input
transformer = transforms.Compose([
        transforms.RandomRotation(10),  # Augmentation: slight rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Augmentation: random shifts
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization
    ])

@app.route('/predict', methods=['POST'])
def predict_digit():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGBA")

    # Flatten alpha channel to white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white_bg, image).convert("L")

    # Resize & pad like MNIST
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_tensor = transformer(img=image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    print(f"Predicted {pred} with confidence {confidence:.2f}")

    return jsonify({
        'prediction': pred,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
