### **Assignment 1: Group Discussion**

**Objective:**  
Review the provided material and build a clear conceptual understanding of unsupervised learning and clustering techniques.

**Tasks:**
- **Read & Highlight:** Skim through the text and highlight the key terms and definitions (e.g., Unsupervised Learning, K-Means, DBSCAN, Elbow Method, Silhouette Score, etc.).
- **Concept Map:** As a group, review:
    - The main ideas of unsupervised learning.
    - How clustering fits into the broader picture.
    - A brief comparison of K-Means versus DBSCAN (including strengths, limitations, and real-world applications).
- **Discussion Points:** Identify at least three real-world scenarios where clustering would be valuable.

---

### **Assignment 2: Implementing K-Means Clustering**

**Objective:**  
Gain hands-on experience with K-Means clustering by working with a synthetic dataset and evaluating cluster quality.

**Tasks:**
1. **Data Generation:**
    - Use Python (and libraries like NumPy and Scikit-Learn) to generate a synthetic dataset (e.g., using `make_blobs`).
2. **Basic K-Means Implementation:**
    - Implement K-Means clustering (or use Scikit-Learn’s `KMeans`).
    - Plot the data points with their assigned clusters and mark the centroids.
3. **Optimal *k* Selection:**
    - Compute and plot the **inertia** (Elbow Method) for a range of *k* values.
    - Calculate the **silhouette score** for each *k* and plot these scores.
    - Discuss (briefly in comments or a markdown cell) which value of *k* seems optimal based on your plots.
4. **Reflection:**
    - Discuss in your code/comments what challenges you might face if the dataset had non-spherical clusters or outliers.

**Deliverable:**  
A Python notebook (Jupyter Notebook or similar) that includes:
- The code for generating the dataset, running K-Means, and plotting the results.
- Plots for the Elbow Method and silhouette scores.
- A short written reflection on your findings.

---

### **Assignment 3: Image Segmentation with K-Means**

**Objective:**  
Apply K-Means clustering to a real-world task—segmenting an image based on color similarity.

**Tasks:**
1. **Load an Image:**
    - Use an image (choose one) and load it using libraries such as `matplotlib.image` or `imageio`.
2. **Preprocess the Image:**
    - Reshape the image array so that each pixel’s RGB values are a data point.
3. **Apply K-Means:**
    - Run K-Means on the pixel data (suggest using *k* = 8, but feel free to experiment).
    - Replace each pixel’s color with its cluster centroid.
4. **Visualize:**
    - Display the original image and the segmented image side by side.
5. **Discussion:**
    - Briefly explain how changing the number of clusters (*k*) might affect the segmentation.

**Deliverable:**  
An updated Python notebook (or a separate notebook) containing:
- The code for loading, processing, and segmenting the image.
- Before/after images.
- A brief comment or markdown cell discussing your observations.

---

### **Assignment 4: Exploring DBSCAN**

**Objective:**  
Learn to use DBSCAN for clustering data with irregular shapes and understand how parameter tuning affects results.

**Tasks:**
1. **Dataset Preparation:**
    - Generate or load a “moons” dataset (using `make_moons` from Scikit-Learn is recommended).
2. **DBSCAN Implementation:**
    - Apply DBSCAN clustering using an initial set of parameters (e.g., `eps=0.05` and `min_samples=5`).
    - Visualize the clusters and identify any noise points.
3. **Parameter Tuning:**
    - Experiment by increasing `eps` (for example, try `eps=0.2`) and observe how the clustering outcome changes.
    - Optionally, vary `min_samples` and note the effect on noise detection.
4. **Reflection:**
    - Write a short paragraph (or include comments) comparing your experience with DBSCAN to K-Means. What kinds of data might favor DBSCAN over K-Means?

**Deliverable:**  
A Python notebook (can be in the same file as previous assignments) with:
- Code for generating the moons dataset and applying DBSCAN.
- Plots showing the clustering results under different parameter settings.
- A brief discussion summarizing your findings.

---

### **General Tips:** 

- **Time Management:** If you find one assignment taking too long, focus on the key parts that demonstrate your understanding of the concepts.
- **Documentation:** Use markdown cells to add explanations and your reasoning behind parameter choices and observations.