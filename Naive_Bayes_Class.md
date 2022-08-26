**CONTENT**

- [Naive Bayes Classifier](#naive-bayes-classifier)
  - [Defination](#defination)
- [K-Means Clustering](#k-means-clustering)
  - [Defination](#defination-1)
- [Naïve Bayes Classifier Algorithm](#naïve-bayes-classifier-algorithm)
  - [Why is it called Naïve Bayes?](#why-is-it-called-naïve-bayes)
  - [Bayes&#39; Theorem:](#bayes-theorem)
  - [Working of Naïve Bayes&#39; Classifier:](#working-of-naïve-bayes-classifier)
    - [Advantages of Naïve Bayes Classifier:](#advantages-of-naïve-bayes-classifier)
    - [Disadvantages of Naïve Bayes Classifier:](#disadvantages-of-naïve-bayes-classifier)
    - [Applications of Naïve Bayes Classifier:](#applications-of-naïve-bayes-classifier)
  - [Types of Naïve Bayes Model:](#types-of-naïve-bayes-model)
  - [Python Implementation of the Naïve Bayes algorithm:](#python-implementation-of-the-naïve-bayes-algorithm)
    - [Steps to implement:](#steps-to-implement)
    - [Data Pre-processing step:](#data-pre-processing-step)
    - [Reference-Naive:](#reference-naive)
  - [What is K-Means Algorithm?](#what-is-k-means-algorithm)
  - [How does the K-Means Algorithm Work?](#how-does-the-k-means-algorithm-work)
  - [How to choose the value of &#34;K number of clusters&#34; in K-means Clustering?](#how-to-choose-the-value-of-k-number-of-clusters-in-k-means-clustering)
    - [Elbow Method](#elbow-method)
  - [Python Implementation of K-means Clustering Algorithm](#python-implementation-of-k-means-clustering-algorithm)
    - [Reference:](#reference)

# Naive Bayes Classifier

### Defination

Naive Bayes Classifier is  **one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions** . It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

# K-Means Clustering

### Defination

K-Means Clustering is an unsupervised learning algorithm that is used to solve the clustering problems in machine learning or data science. In this topic, we will learn what is K-means clustering algorithm, how the algorithm works, along with the Python implementation of k-means clustering.

# Naïve Bayes Classifier Algorithm

* Naïve Bayes algorithm is a supervised learning algorithm, which is based on **Bayes theorem** and used for solving classification problems.
* It is mainly used in *text classification* that includes a high-dimensional training dataset.
* Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
* **It is a probabilistic classifier, which means it predicts on the basis of the probability of an object** .
* Some popular examples of Naïve Bayes Algorithm are  **spam filtration, Sentimental analysis, and classifying articles** .

## Why is it called Naïve Bayes?

The Naïve Bayes algorithm is comprised of two words Naïve and Bayes, Which can be described as:

* **Naïve** : It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.
* **Bayes** : It is called Bayes because it depends on the principle of [Bayes&#39; Theorem](https://www.javatpoint.com/bayes-theorem-in-artifical-intelligence).

## Bayes' Theorem:

* Bayes' theorem is also known as **Bayes' Rule** or  **Bayes' law** , which is used to determine the probability of a hypothesis with prior knowledge. It depends on the conditional probability.
* The formula for Bayes' theorem is given as:

![Naive Bayes Classifier Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/naive-bayes-classifier-algorithm.png)

**Where,**

 **P(A|B) is Posterior probability** : Probability of hypothesis A on the observed event B.

 **P(B|A) is Likelihood probability** : Probability of the evidence given that the probability of a hypothesis is true.

 **P(A) is Prior Probability** : Probability of hypothesis before observing the evidence.

 **P(B) is Marginal Probability** : Probability of Evidence.

## Working of Naïve Bayes' Classifier:

Working of Naïve Bayes' Classifier can be understood with the help of the below example:

Suppose we have a dataset of **weather conditions** and corresponding target variable " **Play** ". So using this dataset we need to decide that whether we should play or not on a particular day according to the weather conditions. So to solve this problem, we need to follow the below steps:

1. Convert the given dataset into frequency tables.
2. Generate Likelihood table by finding the probabilities of given features.
3. Now, use Bayes theorem to calculate the posterior probability.

 **Problem** : If the weather is sunny, then the Player should play or not?

 **Solution** : To solve this, first consider the below dataset:

|              | Outlook  | Play |
| ------------ | -------- | ---- |
| **0**  | Rainy    | Yes  |
| **1**  | Sunny    | Yes  |
| **2**  | Overcast | Yes  |
| **3**  | Overcast | Yes  |
| **4**  | Sunny    | No   |
| **5**  | Rainy    | Yes  |
| **6**  | Sunny    | Yes  |
| **7**  | Overcast | Yes  |
| **8**  | Rainy    | No   |
| **9**  | Sunny    | No   |
| **10** | Sunny    | Yes  |
| **11** | Rainy    | No   |
| **12** | Overcast | Yes  |
| **13** | Overcast | Yes  |

**Frequency table for the Weather Conditions:**

| Weather  | Yes | No |
| -------- | --- | -- |
| Overcast | 5   | 0  |
| Rainy    | 2   | 2  |
| Sunny    | 3   | 2  |
| Total    | 10  | 5  |

**Likelihood table weather condition:**

| Weather  | No        | Yes        |            |
| -------- | --------- | ---------- | ---------- |
| Overcast | 0         | 5          | 5/14= 0.35 |
| Rainy    | 2         | 2          | 4/14=0.29  |
| Sunny    | 2         | 3          | 5/14=0.35  |
| All      | 4/14=0.29 | 10/14=0.71 |            |

**Applying Bayes'theorem:**

**P(Yes|Sunny)= P(Sunny|Yes)*P(Yes)/P(Sunny)**

P(Sunny|Yes)= 3/10= 0.3

P(Sunny)= 0.35

P(Yes)=0.71

So P(Yes|Sunny) = 0.3*0.71/0.35= **0.60**

**P(No|Sunny)= P(Sunny|No)*P(No)/P(Sunny)**

P(Sunny|NO)= 2/4=0.5

P(No)= 0.29

P(Sunny)= 0.35

So P(No|Sunny)= 0.5*0.29/0.35 = **0.41**

So as we can see from the above calculation that **P(Yes|Sunny)>P(No|Sunny)**

**Hence on a Sunny day, Player can play the game.**

### Advantages of Naïve Bayes Classifier:

* Naïve Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
* It can be used for Binary as well as Multi-class Classifications.
* It performs well in Multi-class predictions as compared to the other Algorithms.
* It is the most popular choice for  **text classification problems** .

### Disadvantages of Naïve Bayes Classifier:

* Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

### Applications of Naïve Bayes Classifier:

* It is used for  **Credit Scoring** .
* It is used in  **medical data classification** .
* It can be used in **real-time predictions** because Naïve Bayes Classifier is an eager learner.
* It is used in Text classification such as **Spam filtering** and  **Sentiment analysis** .

## Types of Naïve Bayes Model:

There are three types of Naive Bayes Model, which are given below:

* **Gaussian** : The Gaussian model assumes that features follow a normal distribution. This means if predictors take continuous values instead of discrete, then the model assumes that these values are sampled from the Gaussian distribution.
* **Multinomial** : The Multinomial Naïve Bayes classifier is used when the data is multinomial distributed. It is primarily used for document classification problems, it means a particular document belongs to which category such as Sports, Politics, education, etc.
  The classifier uses the frequency of words for the predictors.
* **Bernoulli** : The Bernoulli classifier works similar to the Multinomial classifier, but the predictor variables are the independent Booleans variables. Such as if a particular word is present or not in a document. This model is also famous for document classification tasks.

## Python Implementation of the Naïve Bayes algorithm:

Now we will implement a Naive Bayes Algorithm using Python. So for this, we will use the " **user_data** "  **dataset** , which we have used in our other classification model. Therefore we can easily compare the Naive Bayes model with the other models.

### Steps to implement:

* Data Pre-processing step
* Fitting Naive Bayes to the Training set
* Predicting the test result
* Test accuracy of the result(Creation of Confusion matrix)
* Visualizing the test set result.

### Data Pre-processing step:

In this step, we will pre-process/prepare the data so that we can use it efficiently in our code. It is similar as we did in [data-pre-processing](https://www.javatpoint.com/data-preprocessing-machine-learning).

### Reference-Naive:

Copied From;
[For More Information Visit this Website(Click Here.)](https://www.javatpoint.com/machine-learning-naive-bayes-classifier#:~:text=Na%C3%AFve%20Bayes%20Classifier%20is%20one,the%20probability%20of%20an%20object.)

## What is K-Means Algorithm?

K-Means Clustering is an [Unsupervised Learning algorithm](https://www.javatpoint.com/unsupervised-machine-learning), which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.

> It is an iterative algorithm that divides the unlabeled dataset into k different clusters in such a way that each dataset belongs only one group that has similar properties.

It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without the need for any training.

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters. The value of k should be predetermined in this algorithm.

The K-means [Clustering](https://www.javatpoint.com/clustering-in-machine-learning) algorithm mainly performs two tasks:

* Determines the best value for K center points or centroids by an iterative process.
* Assigns each data point to its closest k-center. Those data points which are near to the particular k-center, create a cluster.

Hence each cluster has datapoints with some commonalities, and it is away from other clusters.

The below diagram explains the working of the K-means Clustering Algorithm:

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png)

## How does the K-Means Algorithm Work?

The working of the K-Means algorithm is explained in the below steps:

**Step-1:** Select the number K to decide the number of clusters.

**Step-2:** Select random K points or centroids. (It can be other from the input dataset).

**Step-3:** Assign each data point to their closest centroid, which will form the predefined K clusters.

**Step-4:** Calculate the variance and place a new centroid of each cluster.

**Step-5:** Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

**Step-6:** If any reassignment occurs, then go to step-4 else go to FINISH.

**Step-7** : The model is ready.

Let's understand the above steps by considering the visual plots:

Suppose we have two variables M1 and M2. The x-y axis scatter plot of these two variables is given below:

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning2.png)

* Let's take number k of clusters, i.e., K=2, to identify the dataset and to put them into different clusters. It means here we will try to group these datasets into two different clusters.
* We need to choose some random k points or centroid to form the cluster. These points can be either the points from the dataset or any other point. So, here we are selecting the below two points as k points, which are not the part of our dataset. Consider the below image:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning3.png)
* Now we will assign each data point of the scatter plot to its closest K-point or centroid. We will compute it by applying some mathematics that we have studied to calculate the distance between two points. So, we will draw a median between both the centroids. Consider the below image:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning4.png)

From the above image, it is clear that points left side of the line is near to the K1 or blue centroid, and points to the right of the line are close to the yellow centroid. Let's color them as blue and yellow for clear visualization.

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning5.png)

* As we need to find the closest cluster, so we will repeat the process by choosing  **a new centroid** . To choose the new centroids, we will compute the center of gravity of these centroids, and will find new centroids as below:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning6.png)
* Next, we will reassign each datapoint to the new centroid. For this, we will repeat the same process of finding a median line. The median will be like below image:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning7.png)

From the above image, we can see, one yellow point is on the left side of the line, and two blue points are right to the line. So, these three points will be assigned to new centroids.

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning8.png)

As reassignment has taken place, so we will again go to the step-4, which is finding new centroids or K-points.

* We will repeat the process by finding the center of gravity of centroids, so the new centroids will be as shown in the below image:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning9.png)
* As we got the new centroids so again will draw the median line and reassign the data points. So, the image will be:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning10.png)
* We can see in the above image; there are no dissimilar data points on either side of the line, which means our model is formed. Consider the below image:
  ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning11.png)

As our model is ready, so we can now remove the assumed centroids, and the two final clusters will be as shown in the below image:

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning12.png)

## How to choose the value of "K number of clusters" in K-means Clustering?

The performance of the K-means clustering algorithm depends upon highly efficient clusters that it forms. But choosing the optimal number of clusters is a big task. There are some different ways to find the optimal number of clusters, but here we are discussing the most appropriate method to find the number of clusters or value of K. The method is given below:

### Elbow Method

The Elbow method is one of the most popular ways to find the optimal number of clusters. This method uses the concept of WCSS value. **WCSS** stands for  **Within Cluster Sum of Squares** , which defines the total variations within a cluster. The formula to calculate the value of WCSS (for 3 clusters) is given below:

WCSS = ∑~P~i in Cluster1 distance(P~i~ C ~1~)^2^ +∑~P~i in Cluster2 distance(P~i~ C ~2~ ) ^2^ +∑~P~i in CLuster3 distance(P~i~ C ~3~ )^2^

In the above formula of WCSS,

∑~P~i in Cluster1 distance(P~i~ C ~1~ ) ^2^ : It is the sum of the square of the distances between each data point and its centroid within a cluster1 and the same for the other two terms.

To measure the distance between data points and centroid, we can use any method such as Euclidean distance or Manhattan distance.

To find the optimal value of clusters, the elbow method follows the below steps:

* It executes the K-means clustering on a given dataset for different K values (ranges from 1-10).
* For each value of K, calculates the WCSS value.
* Plots a curve between calculated WCSS values and the number of clusters K.
* The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K.

Since the graph shows the sharp bend, which looks like an elbow, hence it is known as the elbow method. The graph for the elbow method looks like the below image:

![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning13.png)

**Note:** We can choose the number of clusters equal to the given data points. If we choose the number of clusters equal to the data points, then the value of WCSS becomes zero, and that will be the endpoint of the plot.

## Python Implementation of K-means Clustering Algorithm

In the above section, we have discussed the K-means algorithm, now let's see how it can be implemented using [Python](https://www.javatpoint.com/python-tutorial).

Before implementation, let's understand what type of problem we will solve here. So, we have a dataset of  **Mall_Customers** , which is the data of customers who visit the mall and spend there.

In the given dataset, we have **Customer_Id, Gender, Age, Annual Income ($), and Spending Score** (which is the calculated value of how much a customer has spent in the mall, the more the value, the more he has spent). From this dataset, we need to calculate some patterns, as it is an unsupervised method, so we don't know what to calculate exactly.

The steps to be followed for the implementation are given below:

* **Data Pre-processing**
* **Finding the optimal number of clusters using the elbow method**
* **Training the K-means algorithm on the training dataset**
* **Visualizing the clusters**

### Reference:

Copied From;
[For More Information Visit this Website(Click Here.)](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)
