# Assignment: Linear Regression on the California Housing Dataset

## Objective

In this assignment, you will use the [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) dataset to build, evaluate, and analyze a linear regression model. You will explore real-world data, preprocess it, train a model, and assess its performance using common metrics. Finally, you will reflect on the results and discuss the model’s strengths and limitations.

> **Optional Extension:**  
> If you’re interested in exploring a different dataset, you may also work with the [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) dataset on Kaggle. This dataset offers additional challenges and opportunities to apply more advanced regression techniques.

---

## Tasks

### 1. Load and Explore the Dataset

- **Data Acquisition:**  
  Use Scikit-Learn’s `fetch_california_housing` to load the dataset into a Pandas DataFrame.

- **Data Exploration:**
    - Print the first few rows of the dataset.
    - Generate summary statistics using `describe()`.
    - Create visualizations (such as pairplots) for key features (e.g., `MedInc`, `AveRooms`, `HouseAge`) and the target variable (`MedHouseVal`).

### 2. Data Preprocessing

- **Feature-Target Split:**  
  Separate the dataset into features ($X$) and the target variable ($y$, which is the median house value).

- **Train/Test Split:**  
  Split the data into training and test sets (e.g., 80% training, 20% testing) using `train_test_split`.

- **Optional:**  
  If you’re interested, apply feature scaling (normalization or standardization) to the data. Consider how this might affect performance if you were using an iterative training method like gradient descent.

### 3. Model Training

- **Build the Model:**  
  Instantiate and train a linear regression model using Scikit-Learn’s `LinearRegression`.

- **Examine the Model:**
    - Print the model’s intercept and coefficients.
    - Reflect on how these coefficients relate to the features in predicting the median house value.

### 4. Model Evaluation

- **Predictions:**  
  Use the trained model to predict house values on the test set.

- **Performance Metrics:**  
  Calculate the Mean Squared Error (MSE) and the coefficient of determination ($R^2$ score) on the test set.

    - **MSE:**  
      $$
      \text{MSE} = \frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
      $$

    - **R² Score:**  
      This measures how well the model explains the variability in the data.

- **Visualization:**  
  Create a scatter plot comparing the actual median house values to the predicted values. Include a reference line where predictions equal the actual values (i.e., the 45° line).

### 5. Discussion and Analysis

Answer the following discussion questions in a brief written report:

1. **Data Preprocessing:**
    - How might feature scaling impact the performance of your model if you were using gradient descent?
    - Why is it important to split the data into training and test sets?

2. **Model Evaluation:**
    - What do the MSE and $R^2$ score tell you about your model’s performance?
    - Are there any limitations or caveats to using these metrics?

3. **Feature Importance:**
    - Examine the model coefficients. Which features appear to be the most important in predicting the median house value?
    - Discuss any possible reasons why these features might be significant.

4. **Model Limitations:**
    - Given the assumptions of linear regression (e.g., linearity, homoscedasticity, normality of errors), what limitations might your model have when predicting housing prices?
    - Suggest possible strategies to address these limitations (e.g., applying polynomial regression or using regularization techniques).

### 6. (Optional) Interactive Exploration

For further exploration, consider using interactive widgets (e.g., with `ipywidgets`) to experiment with:
- Varying the test set size.
- Adjusting the random state for data splitting.
- Observing how these changes impact performance metrics and prediction plots.

---

## Deliverables

1. A Jupyter Notebook (or equivalent) containing:
    - Code cells with your data loading, preprocessing, model training, and evaluation.
    - Visualizations that clearly illustrate your findings.
    - Markdown cells with your written answers to the discussion questions.

2. A short report summarizing your observations, reflections, and any challenges you encountered during the assignment.

---

By completing this assignment, you will gain hands-on experience with a real-world dataset and reinforce your understanding of linear regression, data preprocessing, model evaluation, and the interpretation of results. Good luck!

---

Feel free to include additional notes or interactive components (such as adjusting hyperparameters using `ipywidgets`) to further explore model behavior.