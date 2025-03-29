from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import io

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        return self.fc1(x)

# Set up Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Transform for MNIST-style input
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
    image = image.resize((20, 20))
    final_image = Image.new("L", (28, 28), 255)
    final_image.paste(image, (4, 4))

    image_tensor = transformer(final_image).unsqueeze(0)

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
