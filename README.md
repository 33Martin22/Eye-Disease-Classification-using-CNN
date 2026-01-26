# Eye Disease Classification 👁️

A deep learning-powered web application for classifying eye diseases from retinal images using EfficientNetB7 architecture.

## Overview

This project uses a convolutional neural network (CNN) to classify retinal images into four categories:
- **Normal** - Healthy eye with no abnormalities
- **Diabetic Retinopathy** - Retinal damage caused by diabetes
- **Cataract** - Clouding of the eye's lens
- **Glaucoma** - Optic nerve damage from intraocular pressure



## Features

- 🔍 Dual eye analysis (left and right eye support)
- 📊 Confidence scores for all disease categories
- 🎯 High accuracy predictions using EfficientNetB7
- 💻 User-friendly web interface built with Streamlit
- ☁️ Automatic model download from Google Drive
- 📱 Responsive design for various screen sizes

## Demo
##Click the link below to interact with the model
https://eye-disease-classification-using-cnn-atbs6tgepmjbvqyxsbxbvt.streamlit.app/   
## Technology Stack

- **Deep Learning Framework**: TensorFlow/Keras
- **Model Architecture**: EfficientNetB7
- **Web Framework**: Streamlit
- **Image Processing**: PIL, NumPy
- **Model Storage**: Google Drive (via gdown)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/33Martin22/eye-disease-classification.git
cd eye-disease-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will automatically download the model from Google Drive on first run.

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit
tensorflow
keras
pillow
numpy
gdown
```

## Usage

1. Launch the application using `streamlit run app.py`
2. Upload a retinal image for the left eye
3. Upload a retinal image for the right eye
4. View the predictions and confidence scores for both eyes
5. Each eye will show probabilities for all four categories

### Supported Image Formats

- JPG/JPEG
- PNG

### Image Requirements

- Images are automatically resized to 224x224 pixels
- RGB color format
- Clear retinal/fundus images recommended for best results

## Model Details

- **Architecture**: EfficientNetB7 (pre-trained on ImageNet)
- **Input Size**: 224 × 224 × 3
- **Output Classes**: 4 (Glaucoma, Normal, Diabetic Retinopathy, Cataract)
- **Training Accuracy**: 95%
- **Preprocessing**: EfficientNet-specific preprocessing

## Project Structure

```
eye-disease-classification/
│
├── app.py                          # Main Streamlit application
├── model_EfficientNetB7.h5        # Trained model 
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## How It Works

1. **Image Upload**: User uploads retinal images for both eyes
2. **Preprocessing**: Images are resized to 224×224 and preprocessed using EfficientNet's preprocessing function
3. **Prediction**: The model outputs probability scores for each of the 4 disease categories
4. **Results Display**: The app shows the predicted class and confidence scores for both eyes

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It is **NOT** a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and treatment.

## Future Improvements

- [ ] Add support for more eye diseases
- [ ] Include explainability features (Grad-CAM visualization)
- [ ] Deploy to cloud platforms (Streamlit Cloud, Heroku, AWS)
- [ ] Add batch processing for multiple patients
- [ ] Implement user authentication and history tracking
- [ ] Add PDF report generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EfficientNet architecture by Google Research
- Dataset: Kaggle Datasets
- Streamlit for the amazing web framework
- TensorFlow/Keras teams for deep learning tools

## Contact

Your Name - Martin Kioko kiokomartin27@gmail.com

