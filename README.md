# Temperature Anomaly Detection

This repository contains code and resources for a temperature anomaly detection project using machine learning and Flask.

## Project Overview

- **Machine Learning Model**: Utilizes pre-trained `.pkl` files to predict temperature anomalies and uncertainties.
- **API Integration**: Integrates with the Gemini API to generate insightful descriptions based on the model's predictions.
- **Web Application**: Built with Flask to provide an interactive interface for users to input data and view results.

## Features

- Predicts temperature anomalies based on input data.
- Provides detailed descriptions and preventive measures using the Gemini API.

## Repository Structure

```
/Temperature_anomaly
    ├── anomaly_model.pkl             # Machine learning model file (>100 MB)
    ├── templates/
    │   ├── index.html            # Flask template files
    │   ├── result.html
    ├── app.py                    # Flask application script
    ├── .gitattributes                # Git LFS configuration for large files
    ├── .gitignore                    # Specifies files and directories to ignore
    ├── README.md                     # This file
```

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/Temperature_anomaly.git
cd Temperature_anomaly
```

### 2. Install Git LFS

If you haven't installed Git LFS, follow these steps:

```bash
# For Debian/Ubuntu-based systems
sudo apt-get install git-lfs

# For Red Hat-based systems
sudo yum install git-lfs

# For macOS
brew install git-lfs

# Initialize Git LFS
git lfs install
```

### 3. Configure Git LFS

Track large files with Git LFS:

```bash
# Track .pkl files with Git LFS
git lfs track "*.pkl"

# Add the .gitattributes file to the repository
git add .gitattributes
```

### 4. Add and Commit Files

Add and commit files to the repository:

```bash
# Add large files
git add model/anomaly_model.pkl

# Add other files
git add app/templates/
git add app/app.py

# Commit changes
git commit -m "Initial commit with ML model and Flask templates"
```

### 5. Push to GitHub

Push the changes to GitHub:

```bash
# Push the branch to the remote repository
git push -u origin main
```

## Usage

### 1. Set Up Flask Environment

After making a venv ensure you have Flask and any other dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Run the Flask Application

Start the Flask application:

```bash
python app.py
```

Visit `http://localhost:5000` in your web browser to interact with the application.


## Contact

For any questions or issues, please contact [kumar05.rishu@gmail.com](kumar05.rishu@gmail.com).

---
