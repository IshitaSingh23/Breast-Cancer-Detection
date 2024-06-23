# Breast Cancer Detection

## Overview

This project aims to develop a machine learning model to accurately detect breast cancer based on clinical features. Early detection of breast cancer is crucial for improving treatment outcomes and patient survival rates.

## Features

- **Data Preprocessing**:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling using standardization
- **Exploratory Data Analysis (EDA)**:
  - Visualization of feature distributions
  - Correlation analysis
- **Model Building**:
  - Machine Learning Models Used:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Gradient Boosting Classifier
  - Model Training and Hyperparameter Tuning using Cross-Validation
- **Model Evaluation**:
  - Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion Matrix analysis

The Random Forest classifier achieved the highest accuracy of 97%.

## Dataset

The dataset used is the Breast Cancer Wisconsin dataset, which contains features computed from breast cancer digitized images. It includes 569 instances with 30 feature columns and a target column indicating whether the cancer is benign or malignant.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/IshitaSingh23/Breast-Cancer-Detection.git
    cd Breast-Cancer-Detection
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

4. **View the application**:
    Open your web browser and navigate to `http://localhost:8501`.

## Usage

1. **Upload a CSV file** containing the breast cancer data.
2. **Preprocess the data** using the provided steps.
3. **Extract features** and train the model using the selected classifiers.
4. **Classify new data** to detect whether it is benign or malignant.

## File Structure

- `app.py`: The main file to run the Streamlit application.
- `data/`: Directory containing dataset files.
- `models/`: Directory containing saved models.
- `notebooks/`: Jupyter notebooks for EDA and model training.
- `requirements.txt`: List of required Python packages.
- `README.md`: This readme file.

## Dependencies

- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the developers of the libraries used in this project.
- Special thanks to the contributors and maintainers of Streamlit and Heroku.
