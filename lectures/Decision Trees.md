# **Decision Trees: An In-depth Overview**

## **Table of Contents**
1. [Introduction & Learning Objectives](#introduction--learning-objectives)
2. [What Are Decision Trees?](#what-are-decision-trees)
3. [Inference in a Decision Tree](#inference-in-a-decision-tree)
4. [Growing a Decision Tree: The Algorithm & Working Principle](#growing-a-decision-tree-the-algorithm--working-principle)
    - [Terminology](#terminology)
    - [The Training Process](#the-training-process)
    - [Split Quality Metrics](#split-quality-metrics)
5. [Step-by-Step Example](#example-applying-the-splitter-on-the-coffee-dataset)
6. [Reflection & Key Takeaways](#reflection--key-takeaways)
7. [Read More](#read-more)

---

## **Introduction & Learning Objectives**

**Decision trees** are a fundamental and **interpretable machine learning algorithm** that excel at analyzing **tabular data**. In this session, you will learn:

- **What decision trees are and how they work.**
- **How to construct a decision tree using a greedy algorithm.**
- **How to evaluate splits using measures like Gini impurity and information gain.**
- **How to apply these concepts to a practical example.**

By the end of this document, you should be able to explain the inner workings of a decision tree, understand its training process, and analyze a real-world example.

---

## **What Are Decision Trees?**

Decision trees are a family of **interpretable machine learning algorithms** used for:

- **Classification**
- **Regression**
- **Ranking**
- **Anomaly detection**
- **Uplift modeling**

A **decision tree** is composed of a series of **"questions"**—also known as **conditions**, **splits**, or **tests**—organized in a hierarchical structure. Each **non-leaf node** poses a condition about a feature, and each **leaf node** contains a **prediction** (either a class label or a numerical value).

> **Note:** Unlike botanical trees (where the **root** is at the bottom), decision trees are typically represented with the **root at the top**.

![decision_tree_1](https://developers.google.com/static/machine-learning/decision-forests/images/DecisionTree.png)  
*Figure: A simple classification decision tree. The legend in green is not part of the decision tree.*

---

## **Inference in a Decision Tree**

Inference is the process by which a decision tree makes a prediction. An input example is routed from the **root** through various **conditions** until it reaches a **leaf node**, where the prediction is produced. The sequence of nodes traversed is called the **inference path**.

**Example:**  
Consider the following feature values:

| **num_legs** | **num_eyes** |
|--------------|--------------|
| 4            | 2            |

The decision tree might evaluate:
1. **num_legs ≥ 3** → **Yes**
2. **num_eyes ≥ 3** → **No**

Thus, the prediction is **dog**.

![decision_tree_2](https://developers.google.com/static/machine-learning/decision-forests/images/DecisionTreeInferencePath.png)  
*Figure: The inference path culminating in the leaf **dog** for the example {num_legs: 4, num_eyes: 2}.*

For **regression** tasks, the leaves output numerical predictions. For example, a decision tree might predict a **cuteness score** between 0 and 10.

![decision_tree_3](https://developers.google.com/static/machine-learning/decision-forests/images/DecisionTreeRegression.png)  
*Figure: A decision tree that makes a numerical prediction.*

---

## **Growing a Decision Tree: The Algorithm & Working Principle**

### **Terminology**
- **Nodes:** Decision points where the tree asks a question about a feature.
- **Branches:** Outcomes of a condition that lead to subsequent nodes or final predictions.
- **Leaves:** Terminal nodes containing the final **predictions** (class labels or numerical values).

### **The Training Process**

Training a decision tree involves building a structure that best explains the training data. Since finding the **optimal tree** is **NP-hard** (computationally intractable), heuristic methods (i.e., greedy algorithms) are employed to create a **near-optimal** tree. The process is as follows:

1. **Start at the Root Node:**
   - Begin with the entire dataset.
   - Compute an impurity measure (such as **Gini impurity** or **entropy**) for the target variable.

2. **Consider Candidate Splits:**
   - For each feature, generate candidate splits.
     - **For numerical features:** Sort the values and consider thresholds (often midpoints between unique values).
     - **For categorical features:** Consider splitting by each unique value or grouping similar values.

3. **Evaluate Split Quality:**
   - Divide the dataset into two subsets:
     - **Subset A:** Examples that satisfy the condition.
     - **Subset B:** Examples that do not.
   - Compute the impurity for each subset.
   - Calculate the **information gain** (i.e., the reduction in impurity).
4. **Select the Best Split:**
   - Choose the split that maximizes **information gain** (or minimizes the weighted impurity).

5. **Create a Node and Recurse:**
   - Create a node based on the selected split.
   - Recursively apply the same process to each child node.
   - Continue until a stopping criterion is met (e.g., maximum depth, minimum number of examples, or complete purity).

6. **Finalize the Tree:**
   - The result is a hierarchical structure with **leaf nodes** providing the final predictions.

### **Split Quality Metrics**

- **Gini Impurity:**  
  Measures how often a randomly chosen element would be mislabeled if labeled according to the subset's label distribution.  
  - **Intuitive Explanation:** Lower **Gini impurity** means that a branch is dominated by one class.

- **Information Gain:**  
  Measures the reduction in **entropy** (uncertainty) after the split.  
  - **Intuitive Explanation:** A “good” split results in branches where most examples belong to a single class, thereby reducing uncertainty.

---

## **Step by step Example**

### **The Coffee Dataset**

Consider the following dataset:

| **Time Of Day** | **Tired** | **Drink Coffee** |
|---------------|-----------|-----------------|
| Morning       | Yes       | Yes             |
| Morning       | No        | No              |
| Morning       | Yes       | Yes             |
| Morning       | No        | No              |
| Afternoon     | Yes       | No              |
| Afternoon     | No        | No              |
| Afternoon     | Yes       | No              |
| Afternoon     | No        | No              |

In this dataset, a person **drinks coffee** only when it is **Morning** and the person is **Tired**. Otherwise, no coffee is consumed.

### **Step-by-Step Application**

1. **Evaluate Candidate Splits:**
   - **Splitting on TimeOfDay:**
     - **Morning:** Contains 4 examples. In the Morning group:
       - If **Tired = Yes:** **DrinkCoffee = Yes**
       - If **Tired = No:** **DrinkCoffee = No**
       - *Result:* Mixed group (2 Yes, 2 No).
     - **Afternoon:** Contains 4 examples. Regardless of tiredness, **DrinkCoffee = No**.
       - *Result:* Pure group (all No).
   - **Splitting on Tired:**
     - **Tired = Yes:** Contains 4 examples (2 from Morning, 2 from Afternoon).  
       - Morning examples: **DrinkCoffee = Yes**  
       - Afternoon examples: **DrinkCoffee = No**  
       - *Result:* Mixed group (2 Yes, 2 No).
     - **Tired = No:** Contains 4 examples (all with **DrinkCoffee = No**).
       - *Result:* Pure group.
   - Both splits yield one pure branch and one impure branch. In this case, the algorithm may choose either. For this example, we assume the algorithm picks **TimeOfDay** as the first split.

2. **Construct the Decision Tree:**
   - **First Split:** Check **TimeOfDay**.
     - **If TimeOfDay = Morning:** Further split on **Tired**.
     - **If TimeOfDay = Afternoon:** The branch is pure with **DrinkCoffee = No**.
   - **Second Split (Morning Branch):**
     - **If Tired = Yes:** Predict **DrinkCoffee = Yes**.
     - **If Tired = No:** Predict **DrinkCoffee = No**.

Thus, the resulting decision tree might look like:

```
           [Is Time Of Day = Morning?]
               /                 \
            Yes                   No
         [Is Tired?]            (Drink Coffee = No)
           /     \
        Yes      No
 (Drink Coffee = Yes) (Drink Coffee = No)
```

---

## **Reflection & Key Takeaways**

- **Interpretable Model:** Decision trees offer a clear, hierarchical structure that mirrors human decision-making.
- **Greedy Algorithm:** Trees are grown using a heuristic approach that selects splits based on **information gain** or reduction in **impurity**.
- **Effective Splits:** **Gini impurity** and **information gain** are key metrics for evaluating split quality.
- **Practical Application:** The updated coffee dataset example shows how both **TimeOfDay** and **Tired** are used to predict whether a person drinks coffee.

**Reflective Questions:**
1. What challenges might arise when the features interact (e.g., a feature is only predictive in combination with another)?
2. How could overfitting be mitigated in decision tree models?
3. Under what circumstances would ensemble methods (like **Random Forests**) be more effective than a single decision tree?

---

## **Read More**

For additional insights into decision trees and related topics, please explore the following resources:

- [**Decision Trees (Wikipedia)**](https://en.wikipedia.org/wiki/Decision_tree)
- [**Random Forests (Wikipedia)**](https://en.wikipedia.org/wiki/Random_forest)
- [**Information Gain (Wikipedia)**](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
- [**Gini Impurity (Wikipedia)**](https://en.wikipedia.org/wiki/Gini_coefficient)
- [**Supervised Learning (Wikipedia)**](https://en.wikipedia.org/wiki/Supervised_learning)
- [**Machine Learning Crash Course: Accuracy, Precision, and Recall (Google Developers)**](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)