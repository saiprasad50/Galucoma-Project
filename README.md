# Glaucoma Detection Web Application

This web application uses a deep learning model to detect glaucoma from eye fundus images.

## Features

- Upload and analyze eye fundus images
- Real-time glaucoma detection using CNN model
- Modern 3D UI interface
- Supports multiple image formats (PNG, JPG, JPEG)

## Project Structure

```
final_code/
    ├── app.py              # Main Flask application
    ├── g.py               # Helper functions
    ├── templates/         # HTML templates
    │   └── index.html    # Main page template
    ├── data/             # Training and validation datasets (not included in repo)
    ├── uploads/          # Temporary folder for uploaded images
    └── glaucoma_cnn_model.h5  # Trained model file (not included in repo)
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install flask tensorflow numpy pillow
   ```
3. Download the model file and place it in the root directory
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Open the web application in your browser
2. Upload an eye fundus image
3. Click "Upload & Detect" to analyze the image
4. View the detection results

## Note

The trained model file (`glaucoma_cnn_model.h5`) and dataset are not included in this repository due to size constraints. Please contact the maintainers for access to these files.