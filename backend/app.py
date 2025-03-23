from flask import Flask,request,jsonify,send_from_directory
from flask_cors import CORS
import torch

app = Flask(__name__)

model = torch.load('model.pth')

@app.route('/')
def main_app():
    return send_from_directory(app.static_folder,'index.xhtml')

