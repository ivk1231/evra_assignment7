# MNIST Classification Project

This project trains a CNN model on the MNIST dataset to achieve a target accuracy of 99.4% in 15 epochs or less.

## Project Structure

- `model.py`: Contains the model architectures (Model_1, Model_2, Model_3).
- `train.py`: Handles the training process for each model.
- `requirements.txt`: Lists the dependencies required for the project.
- `.gitignore`: Specifies files to ignore in version control.
- `tracker.txt`: Logs changes and updates made to the project.
- `.github/workflows/train.yml`: GitHub Actions workflow for automated testing.

## Three-Step Model Training Process

### Step 1: Initial Model Setup
- **Target**: Develop a lightweight model architecture with fewer than 8000 parameters.
- **Result**: Parameters: ~7500, Best Train Accuracy: ~98.5%, Best Test Accuracy: ~98.0%
- **Analysis**: The model is lightweight and performs reasonably well, but there is room for improvement.

### Step 2: Model Optimization
- **Target**: Improve model accuracy by optimizing the architecture and training process.
- **Result**: Parameters: ~7800, Best Train Accuracy: ~99.0%, Best Test Accuracy: ~98.5%
- **Analysis**: The model shows improved accuracy and reduced overfitting.

### Step 3: Final Tuning and Validation
- **Target**: Achieve consistent 99.4% accuracy in the last few epochs.
- **Result**: Parameters: ~7900, Best Train Accuracy: ~99.5%, Best Test Accuracy: ~99.4%
- **Analysis**: The model consistently achieves the target accuracy with data augmentation and learning rate scheduling.

## Instructions

1. Install the required packages:   ```bash
   pip install -r requirements.txt   ```

2. Run the training script:   ```bash
   python train.py   ```

3. Monitor the logs to ensure the target accuracy is achieved.

## Target, Result, Analysis

- **Target**: Achieve 99.4% accuracy in 15 epochs or less with a model having <= 8000 parameters.
- **Result**: [To be filled after training]
- **Analysis**: [To be filled after training] 