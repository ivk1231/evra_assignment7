# MNIST Classification Project

This project trains CNN models on the MNIST dataset to achieve a target accuracy of 99.4% in 15 epochs or less.

## Project Structure

- `model.py`: Contains the model architectures (Model_1, Model_2, Model_3).
- `train.py`: Handles the training process for each model.
- `requirements.txt`: Lists the dependencies required for the project.
- `.gitignore`: Specifies files to ignore in version control.
- `tracker.txt`: Logs changes and updates made to the project.
- `.github/workflows/train.yml`: GitHub Actions workflow for automated testing.

## Three-Step Model Training Process

### Step 1: Initial Model Setup (Model_1)
- **Target**: Develop a lightweight model architecture with fewer than 8000 parameters.
- **Result**: Parameters: ~7500, Best Train Accuracy: 99.43% (Epoch 12)
- **Analysis**: The basic model achieves target accuracy efficiently, reaching 99.43% in 12 epochs with a steady learning curve.

### Step 2: Model Optimization (Model_2)
- **Target**: Improve model accuracy by optimizing the architecture and training process.
- **Result**: Parameters: ~7800, Best Train Accuracy: 99.23% (Epoch 15)
- **Analysis**: Model shows consistent improvement with batch normalization and dropout, though not exceeding Model_1's performance.

### Step 3: Final Tuning (Model_3)
- **Target**: Achieve consistent 99.4% accuracy with enhanced architecture.
- **Result**: Parameters: ~7900, Best Train Accuracy: 99.42% (Epoch 10)
- **Analysis**: Achieved target accuracy fastest among all models, reaching 99.42% in just 10 epochs, demonstrating effective optimization.

## Instructions

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. Monitor the logs to ensure the target accuracy is achieved.

## Target, Result, Analysis

- **Target**: Achieve 99.4% accuracy in 15 epochs or less with models having <= 8000 parameters.
- **Result**: 
  - Model_1: 99.43% in 12 epochs (Loss: 7.9593)
  - Model_2: 99.23% in 15 epochs (Loss: 10.9033)
  - Model_3: 99.42% in 10 epochs (Loss: 7.9667)
- **Analysis**: All models maintained parameter count under 8000 while achieving high accuracy. Model_3 proved most efficient, reaching target accuracy in fewer epochs, while Model_1 achieved the highest overall accuracy. Model_2's regularization techniques provided stable training but slightly lower peak performance. 