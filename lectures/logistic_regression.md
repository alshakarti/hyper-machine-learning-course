# Logistic Regression

## Fundamentals

**Logistic regression** is a type of regression model that predicts a probability. Logistic regression models have the following characteristics:

- **Categorical Labels:**  
  The label is categorical. Typically, logistic regression refers to binary logistic regression, which calculates probabilities for labels with two possible values. A less common variant, multinomial logistic regression, calculates probabilities for labels with more than two possible values.

- **Loss Function:**  
  The loss function during training is *Log Loss*. (For labels with more than two possible values, multiple Log Loss units can be placed in parallel.)

- **Model Architecture:**  
  The model has a linear architecture—not a deep neural network. However, this definition also applies to deep models that predict probabilities for categorical labels.

### Example

Consider a logistic regression model that calculates the probability of an input email being either spam or not spam. Suppose the model predicts **0.72** during inference. Then, the model is estimating:

- A **72%** chance that the email is spam.
- A **28%** chance that the email is not spam.

## Two-Step Architecture

A logistic regression model uses the following two-step architecture:

1. **Linear Prediction:**  
   The model generates a raw prediction ($ y' $) by applying a linear function of the input features.
2. **Sigmoid Transformation:**  
   The raw prediction is passed through a sigmoid function, which converts it to a value between 0 and 1 (exclusive).

Like any regression model, a logistic regression model predicts a number. This number typically becomes part of a binary classification model as follows:

- If the predicted number is greater than the classification threshold, the binary classification model predicts the positive class.
- If the predicted number is less than the classification threshold, the binary classification model predicts the negative class.

---

## Learning Objectives

- Identify use cases for performing logistic regression.
- Explain how logistic regression models use the sigmoid function to calculate probability.
- Compare linear regression and logistic regression.
- Explain why logistic regression uses log loss instead of squared loss.
- Explain the importance of regularization when training logistic regression models.


In the Linear regression session, you explored how to construct a model to make continuous numerical predictions (for example, predicting the fuel efficiency of a car). But what if you want to build a model to answer questions like “Will it rain today?” or “Is this email spam?”

This module introduces a new type of regression model called [**logistic regression**] that is designed to predict the probability of a given outcome.



# Logistic regression: Calculating a probability with the sigmoid function

Many problems require a probability estimate as output. [**Logistic regression**](/machine-learning/glossary#logistic_regression) is an extremely efficient mechanism for calculating probabilities. Practically speaking, you can use the returned probability in either of the following two ways:

- **Applied as is.**: For example, if a spam-prediction model takes an email as input and outputs a value of `0.932`, this implies a **93.2%** probability that the email is spam.
- **Converted to a binary category.**: For example, converting the output into a binary category such as `True` or `False`, `Spam` or `Not Spam`.

This module focuses on using logistic regression model output as-is.

---

## Sigmoid function

You might be wondering how a logistic regression model can ensure its output represents a probability—that is, always output a value between 0 and 1. In fact, there is a family of functions called **logistic functions** whose outputs have those characteristics. The standard logistic function, also known as the [**sigmoid function**](/machine-learning/glossary#sigmoid-function) (*sigmoid* means "s-shaped"), is given by:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Figure 1 shows the corresponding graph of the sigmoid function.

![Sigmoid (s-shaped) curve plotted on the Cartesian coordinate plane, centered at the origin.](https://developers.google.com/static/machine-learning/crash-course/logistic-regression/images/sigmoid_function_with_axes.png)  
**Figure 1.** Graph of the sigmoid function. The curve approaches 0 as *x* values decrease to negative infinity, and 1 as *x* values increase toward infinity.

As the input, `x`, increases the output of the sigmoid function approaches (but never reaches) `1`. Similarly, as the input decreases the output approaches (but never reaches) `0`.

> **Deep dive into the math behind the sigmoid function:**  
> The table below shows the output values of the sigmoid function for input values in the range –7 to 7. Notice how quickly the sigmoid approaches 0 for large negative inputs and 1 for large positive inputs. Regardless of how extreme the input, the output will always be greater than 0 and less than 1.

| Input | Sigmoid output |
|:-----:|:--------------:|
| -7    | 0.001          |
| -6    | 0.002          |
| -5    | 0.007          |
| -4    | 0.018          |
| -3    | 0.047          |
| -2    | 0.119          |
| -1    | 0.269          |
| 0     | 0.50           |
| 1     | 0.731          |
| 2     | 0.881          |
| 3     | 0.952          |
| 4     | 0.982          |
| 5     | 0.993          |
| 6     | 0.997          |
| 7     | 0.999          |

---

### Transforming linear output using the sigmoid function

The following equation represents the linear component of a logistic regression model:

$$
z = b + w_1x_1 + w_2x_2 + \ldots + w_Nx_N
$$

where:

- **$z$** is the output of the linear equation (also called the *log odds*).
- **$b$** is the bias.
- The **$w$** values are the model’s learned weights.
- The **$x$** values are the feature values for a particular example.

To obtain the logistic regression prediction, the value of $z$ is passed through the sigmoid function, yielding a probability between 0 and 1:

$$
y' = \frac{1}{1 + e^{-z}}
$$

> **More about log-odds:**  
 In the equation  
 $$
 z = b + w_1x_1 + w_2x_2 + \ldots + w_Nx_N,
 $$  
 the value $z$ is called the **log-odds** because starting with the sigmoid function  
 $$
 y = \frac{1}{1 + e^{-z}},
 $$  
 solving for $z$ gives  
 $$
 z = \log\left(\frac{y}{1-y}\right).
 $$  
> Here, $z$ is the logarithm of the ratio of the probabilities of the two outcomes: $y$ and $1-y$.

Figure 2 illustrates how the linear output is transformed into the logistic regression output using these calculations.

![Left: Graph of the linear function $z = 2x + 5$ with three points highlighted. Right: Sigmoid curve with the same three points highlighted after being transformed by the sigmoid function.](https://developers.google.com/static/machine-learning/crash-course/logistic-regression/images/linear_to_logistic.png)  
**Figure 2.** Left: Graph of the linear function $z = 2x + 5$ with three points highlighted. Right: Sigmoid curve with the same three points highlighted after transformation.

In Figure 2, a linear equation is transformed by the sigmoid function into an S-shaped curve. Note that while the linear equation can produce very large or very small values of $z$, the output of the sigmoid function $y'$ is always between 0 and 1 (but never exactly 0 or 1). For example, even if a point has $z = -10$, the sigmoid function maps it to a value near 0 (approximately 0.00004).

---

## Exercise: Check your understanding

A logistic regression model with three features has the following bias and weights:

**Parameters:**


| Parameter | Value |
|-----------|-------|
| $b$       | 1     |
| $w_1$     | 2     |
| $w_2$     | -1    |
| $w_3$     | 5     |

---

Given the following input values:


| Feature | Value |
|---------|-------|
| $x_1$   | 0     |
| $x_2$   | 10    |
| $x_3$   | 2     |

---

Answer the following two questions:

1. **What is the value of $z$ for these input values?**

    - A. –1
    - B. 0
    - C. 0.731
    - D. 1

2. **What is the logistic regression prediction for these input values?**

    - A. 0.268
    - B. 0.5
    - C. 0.731
    - D. 1