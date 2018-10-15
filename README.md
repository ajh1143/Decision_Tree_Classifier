# Decision_Tree_Classifier

## Classification and regression trees (CART) Algorithms
CART algorithms are `Supervised` learning models used for problems involving classification and regression.

## Supervised Learning
Supervised learning is an approach for engineering predictive models from **known** labeled data. Our goal is to predict future labels (outputs), based on our features (inputs) when introduce to a novel dataset.

## Classification Example Problems
1)Identifying fake profiles.

2)Classifying a species.

3)Predicting what sport someone plays.

## Classification Tree - Overview
Sequence of if-else questions about features.

Objective is to infer class labels from unseen data.

Capture non-linear relationships between features and labels.

No feature scaling required.

## Classification Model - Approach
Given a dataset, break it into training and test sets. 

We can then use the Training dataset with a learning algorithm (in our case, the scikit-learn DecisionTreeClassifier) to create a model via induction, which is then applied to make predictions on the test set of data through deduction. 

## Model Concerns

Are you selecting the **right problem** to test?

Are you capable of supplying **sufficient data**?

Are you providing your model **clean data**?

Are you able to prevent algorithmic **biases and confounding factors**?

## Performance Metrics

Goal: Achieve the highest possible **accuracy**, while retaining the lowest **error rate**.

`Accuracy`  = `# correct predictions / # total predictions`

`Error rate` = `# number wrong predictions / # total predictions`

## Tree Data Structure - Fundamentals
### Reference Image
<img src="https://github.com/ajh1143/KNN_ModelSelector/blob/master/Images/DTC/tree_anatomy.jpeg" class="inline"/><br>

Unlike common linear data structures, like `lists` and `arrays`, a `Tree` is a non-linear, hierarchical method of storing/modeling data. Visually, you can picture an evolutionary tree, a `document object model (DOM)` from HTML, or even a flow chart of a company hierarchy. In contrast to a biological tree originating of kingdom plantae, the data structure tree has a simple anatomy. 

A tree consists of **nodes** and **edges**. 

There are 'specialized' kinds of nodes we classify by unique names which represent their place on the hierarchy of our structure. The **root** node, is a single point of origin for the rest of the tree. Branches can extend from nodes and link to other nodes, with the link referred to as an **edge**. The node which accepts a link/edge, is said to be a **child** node, and the originating node is the **parent**. A single node may have one, two, or no children. If a node has no children, but does have a parent, it is called a **leaf**. Some also refer to **internal** nodes, which have one parent and two children. Finally, **sibling** nodes are nodes which share a parent. 

Beyond the core anatomy, the tree has unique metrics to be explored: **depth** and **height**. **Depth** refers to the spatial attributes of an individual node in relation to the root, meaning, how many links/edges are between the specific node and the root node. The **height**, refers to the number of edges in the longest possible path of the tree, similar to finding the longest carbon chain back in organic chemistry to determine the IUPAC name of the compound.

## Decision Tree
<img src="https://github.com/ajh1143/KNN_ModelSelector/blob/master/Images/DTC/Decision_tree_anatomy.jpg" class="inline"/><br>
A decision tree, allows us to run a series of **if/elif/else** tests/questions on a data point, record, or observation with many attributes to be tested. Each node of this tree, would represent some condition of an attribute to test, and the edges/links are the results of this test constrained to some kind of binary decision. For example, if we had a dataset with rich features about a human, we could ask many questions about that person and their behavior based on gender, weight, height, activities etc. 

## Classification Plot - Simple
<img src="https://github.com/ajh1143/KNN_ModelSelector/blob/master/Images/DTC/Boundary.jpg" class="inline"/><br>

**Decision Region**: Space where instances become assigned to particular class, blue or red, in the plotting space in the diagram. 

**Decision Boundary**: Point of transition from one decision region to another, aka one class/label to another, the diagonal black line.


## Feature Selection

Identification of which features/columns with the highest weights in predictive power. 

Removing low variance features
`from sklearn.feature_selection import VarianceThreshold`

In our case, the CART algorithm will do the feature selection for us through the `Gini Index` or `Entropy` which measure how pure your data is partitioned through it's journey to a final leaf node classification.  


## Decision Tree Classification: Steps to Build and Run

1 **Imports**

2 **Load Data**

3 **Test and Train Data**

4 **Instantiate a Decision Tree Classifier**

5 **Fit data**

6 **Predict**

7 **Check Accuracy**

## Import SciKit-Learn Modules
```Python3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## Load Dataset

First, we're going to want to load a dataset, and create two sets, X and y, which represent our features and our desired label.

```Python3
# X contains predictors, y holds the classifications
X, y = dataset.data, dataset.target
features = iris.feature_names
```

## Split dataset into Test and Train sets
Now, we can partition our data into test and train sets, and the typical balance is usually 80/20 or 70/30 test vs train percentages. 

The results of the split will be stored in `X_train`, `X_test`, `y_train`, and `y_test`. 

```Python3
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
```

## Instantiate a DecisionTreeClassifier
We can instantiate our DecisionTreeClassifier object with a `max_depth`, and a `random_state`.

`random_state` allows us a way of ensuring reproducibility, while `max_depth` is a **hyper-parameter**, which allows us to control complexity of our tree, but must be used with a cautious awareness. If `max_depth` is set too high, we risk over-fitting the data, while if it's too low, we will be underfitting. 

```Python3
dt = DecisionTreeClassifier(max_depth=6, random_state=1)
```

## Fitting the model
We fit our our model by utilizing `.fit()` and feeding it parameters `X_train` and `y_train` which we created previously.

```Python3
dt.fit(X_train, y_train)
```

## Predict test set labels
We can now test our model by applying it to our `X_test` variable, and we'll store this as `y_pred`. 

```Python3
y_pred = dt.predict(X_test)
```

## Print predictions
```Python3
print(y_pred[0:5])
```

## Check accuracy
We want to check the accuracy of our model, and we can do so simply by calling `accuracy_score()` with `y_test` and `y_pred`.

```Python3
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
```
