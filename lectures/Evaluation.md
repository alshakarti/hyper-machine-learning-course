# **Evaluation Metrics**

When evaluating a decision tree—or any **classification model**—we rely on several key **metrics** calculated from **true positives (TP)**, **false positives (FP)**, **true negatives (TN)**, and **false negatives (FN)**. The choice of metric depends on the model’s purpose, the cost of different misclassifications, and whether the dataset is balanced or imbalanced. Many of these metrics depend on a fixed **classification threshold**, which can be tuned to optimize performance.

---

## **The Confusion Matrix**

A **confusion matrix** is a table used to describe the performance of a classification model by summarizing the counts of correct and incorrect predictions for each class. For example, consider the following confusion matrix:

|                         | **Predicted Positive** | **Predicted Negative** |
|-------------------------|------------------------|------------------------|
| **Actual Positive**     | **TP** (e.g., 40)      | **FN** (e.g., 10)      |
| **Actual Negative**     | **FP** (e.g., 5)       | **TN** (e.g., 45)      |

*Explanation:*  
- **TP (True Positives):** 40 cases where positive instances were correctly predicted.  
- **FN (False Negatives):** 10 cases where positive instances were incorrectly predicted as negative.  
- **FP (False Positives):** 5 cases where negative instances were incorrectly predicted as positive.  
- **TN (True Negatives):** 45 cases where negative instances were correctly predicted.

---

## **Evaluation Metrics Derived from the Confusion Matrix**

Using the values from the confusion matrix above, we can calculate several common **evaluation metrics**:

| **Metric**       | **Formula**                                          | **Example Calculation**                     | **Result**              |
|------------------|------------------------------------------------------|---------------------------------------------|-------------------------|
| **Accuracy**     | (TP + TN) / (TP + FP + TN + FN)                       | (40 + 45) / (40 + 5 + 45 + 10)                | 85 / 100 = **85%**      |
| **Precision**    | TP / (TP + FP)                                       | 40 / (40 + 5)                               | 40 / 45 ≈ **88.9%**     |
| **Recall (TPR)** | TP / (TP + FN)                                       | 40 / (40 + 10)                              | 40 / 50 = **80%**       |
| **FPR**          | FP / (FP + TN)                                       | 5 / (5 + 45)                                | 5 / 50 = **10%**        |

### **Accuracy**

**Accuracy** is the proportion of all predictions (both positive and negative) that are correct. It is defined as:

$$$
\text{Accuracy} = \frac{\textbf{TP} + \textbf{TN}}{\textbf{TP} + \textbf{TN} + \textbf{FP} + \textbf{FN}}
$$$

For example, in a **spam classification** task, accuracy represents the fraction of emails correctly classified as either spam or not spam. While accuracy provides a coarse measure of overall performance, it can be misleading for imbalanced datasets.

### **Recall (True Positive Rate)**

**Recall** (or **True Positive Rate, TPR**) measures the proportion of actual positives that are correctly identified:

$$$
\text{Recall} = \frac{\textbf{TP}}{\textbf{TP} + \textbf{FN}}
$$$

For instance, in **spam filtering**, recall indicates the fraction of spam emails that were correctly flagged. A perfect model would have a recall of **1.0** (or **100%**), meaning every spam email is detected.

### **False Positive Rate (FPR)**

The **False Positive Rate (FPR)** quantifies the proportion of actual negatives that were incorrectly classified as positives:

$$$
\text{FPR} = \frac{\textbf{FP}}{\textbf{FP} + \textbf{TN}}
$$$

In a spam classification scenario, FPR represents the fraction of legitimate emails that were mistakenly marked as spam. A perfect model would have an FPR of **0.0**, indicating no false alarms.

### **Precision**

**Precision** is the proportion of positive predictions that are correct:

$$$
\text{Precision} = \frac{\textbf{TP}}{\textbf{TP} + \textbf{FP}}
$$$

For example, in **spam detection**, precision indicates the fraction of emails classified as spam that are indeed spam. A model with high precision rarely misclassifies legitimate emails as spam.

---

## **Balancing Precision and Recall**

Precision and recall often exhibit an inverse relationship. Adjusting the **classification threshold** can improve one metric at the expense of the other. The following table summarizes these trade-offs:

| **Adjustment**              | **Effect on Precision**                             | **Effect on Recall**                      |
|-----------------------------|-----------------------------------------------------|-------------------------------------------|
| **Increasing the Threshold**| Fewer false positives → **Precision increases**       | More false negatives → **Recall decreases** |
| **Decreasing the Threshold**| More false positives → **Precision decreases**        | Fewer false negatives → **Recall increases** |

*Explanation:*  
- **Increasing the threshold:** The model becomes stricter about labeling an instance as positive, reducing false positives but possibly increasing false negatives.  
- **Decreasing the threshold:** The model becomes more lenient, capturing more positives (improving recall) but also increasing false positives (reducing precision).

By considering these metrics together, you gain a nuanced understanding of your model’s performance—especially when simple accuracy is not enough, such as in cases of imbalanced datasets. Adjusting the threshold to optimize the metric that best reflects your project's goals is an important step in model evaluation.

---

## **Read More**

For additional details on **evaluation metrics**, consider exploring the following resources:  

- [Confusion Matrix (Wikipedia)](https://en.wikipedia.org/wiki/Confusion_matrix)  
- [Precision and Recall (Wikipedia)](https://en.wikipedia.org/wiki/Precision_and_recall)  
- [Accuracy, Precision, and Recall (Google Developers)](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)
- [Classification: ROC and AUC (Google Developers)](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

These articles provide further insight into how these metrics are calculated and used in various machine learning applications.