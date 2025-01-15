# Multi-Label Text Classification using TF-IDF and Logistic Regression

## Task
Classify textual data using a multi-label approach.

## Stack
- **TF-IDF**: For text vectorization.
- **Logistic Regression**: For multi-label classification.

## Steps
1. **Data Loading and Normalization**:
   - Loaded data from a file and cleaned text for better processing.
2. **Label Binarization and Title Vectorization**:
   - Converted text labels to binary format and vectorized titles using TF-IDF.
3. **Data Splitting**:
   - Split the data into training and testing sets with an 80/20 ratio.
4. **Model Training**:
   - Trained the Logistic Regression model using `MultiOutputClassifier`.
5. **Model Evaluation**:
   - Measured model performance using standard metrics such as precision, recall, and F1-score.

## Results
### Accuracy
- **Achieved**: **80%**
- **Comparison**: 9% lower than the Random Forest model.

### Classification Report

| **Label**                   | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------------------------|---------------|------------|--------------|-------------|
| **Vice President**          | 0.91          | 0.75       | 0.82         | 67          |
| **Director**                | 0.94          | 0.87       | 0.90         | 97          |
| **Individual Contributor/Staff** | 0.93  | 0.97       | 0.95         | 226         |
| **Manager**                 | 0.85          | 0.34       | 0.49         | 32          |
| **Chief Officer**           | 0.88          | 0.17       | 0.29         | 40          |
| **Micro Avg**               | 0.93          | 0.80       | 0.86         | 464         |
| **Macro Avg**               | 0.75          | 0.52       | 0.58         | 464         |
| **Weighted Avg**            | 0.91          | 0.80       | 0.83         | 464         |
| **Samples Avg**             | 0.83          | 0.82       | 0.82         | 464         |

---
### Total model accuracy: 0.8013392857142857
## Challenges
1. **Owner Label Issue**:
   - The `Owner` label consistently received a precision, recall, and F1-score of `0.00` due to a lack of examples in the dataset.
2. **Rare Class Handling**:
   - Insufficient attention to rare classes, leading to imbalanced predictions.

---

## Solutions Undertaken
1. Added basic stop-word cleaning for improved text preprocessing.
2. Tested different hyperparameters, such as `max_iter` and `test_size`.
3. Implemented a Random Forest model for comparison, achieving higher accuracy.

---

## Other Optimization Possibilities
1. Improve **vectorization** by exploring tools like `Word2Vec` or `GloVe`.
2. Use **NLTK tools** to clean noise in the text data further.
3. Perform **class balancing** to improve handling of rare metrics like `Owner`.
4. Experiment with **advanced models**:
   - Random Forest
   - Multi-Layer Perceptrons (MLP)
   - Transformer-based models like BERT

---

## Installation and Usage
### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/stiffstifler/TF_IDF_Logistic_Regression.git
   cd TF_IDF_Logistic_Regression
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the script:
   ```bash
   python main.py
   or
   python3 main.py
   ```

## Requirements
The following libraries were used:

- numpy==2.2.1
- pandas==2.2.3
- scikit-learn==1.6.1

## Attention, dataset is missing!