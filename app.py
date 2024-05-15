import os
from flask import Flask, render_template, request
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

# Define the Deepfake detection Model class
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Define the validation_dataset class
class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        vidObj = cv2.VideoCapture(self.video_path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                faces = face_recognition.face_locations(image)
                if faces:  # Proceed only if faces are detected
                    top, right, bottom, left = faces[0]
                    image = image[top:bottom, left:right, :]
                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
                    if len(frames) == self.count:
                        break
        # Pad frames if there are not enough
        while len(frames) < self.count:
            frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

# Load the deepfake detection model
deepfake_model = Model(2).cuda()
path_to_deepfake_model = r"Deep_fake_model.pt"  # Provide the path to the trained deepfake detection model
deepfake_model.load_state_dict(torch.load(path_to_deepfake_model))
deepfake_model.eval()

# Define the audio analysis function
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

# Define the audio analysis function
def analyze_audio(input_audio_path):
    model_filename = r"svm_model.pkl"
    scaler_filename = r"scaler.pkl"

    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(mfcc_features_scaled)

        if prediction[0] == 0:
            return "The input audio is classified as REAL AUDIO."
        else:
            return "The input audio is classified as FAKE AUDIO."
    else:
        return "Error: Unable to process the input audio."

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_type = request.form['type']
        if file:
            if file_type == 'audio':
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(audio_path)
                result = analyze_audio(audio_path)
                os.remove(audio_path) 
                return render_template('audio_result.html', result=result)
            elif file_type == 'video':
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(video_path)

                # Define transformation for video frames
                im_size = 112
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                train_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((im_size, im_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

                # Create dataset
                video_dataset = ValidationDataset(video_path, sequence_length=20, transform=train_transforms)

                # Make deepfake prediction
                prediction = predict(deepfake_model, video_dataset[0], './')
                if prediction[0] == 1:
                    result = "The input video is classified as REAL VIDEO."
                else:
                    result = "The input video is classified as FAKE VIDEO."

                return render_template('video_result.html', result=result)

    return render_template('index.html')

def predict(model, img, path='./'):
    fmap, logits = model(img.to('cuda'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = nn.Softmax(dim=1)(logits)
    confidence = logits[:, 1].item() * 100  # Confidence for class 1 (index 1)
    print('Confidence of prediction:', confidence)
    prediction_idx = torch.argmax(logits, dim=1).item()
    
    return [prediction_idx, confidence]

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
