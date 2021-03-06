Machine learning equips us with powerful statistical insights to make more informed decisions about business and life.

What if you could match your potential customers with the products they want? Or automate the identification of diseases, diagnoses, examinations? Wouldn't it be useful if you could make predictions about something you've never seen using the aggregate wisdom of others?

You can, with the help of a decision tree.

Machine learning reveals a unique view of the data, but you still need to think for yourself.

_______________________________________________________________________________________________________________________________________

## The Agenda

1. Foundations of Data Structures, Decision Trees, Performance Metrics.
2. Steps To Build a Basic Decision Tree
3. Applied Case Study

Today, let’s focus specifically on the classification portion of the CART set of algorithms, by utilizing SciKit-Learn’s Decision Tree Classifier module to build machine learning solutions.

Let's begin by diving into the basics.

_______________________________________________________________________________________________________________________________________

# PART I - FOUNDATIONS    
## Classification and regression trees (CART) Algorithms
**CART** algorithms are **Supervised** learning models used for problems involving **classification** and **regression**.

## Supervised Learning
**Supervised learning** is an approach for engineering **predictive models** from **known** labeled data, meaning the dataset already contains the targets appropriately classed. Our goal is to allow the algorithm to build a model from this known data, to **predict** future **labels** (outputs), based on our **features** (inputs) when introduced to a novel dataset.

## Classification Example Problems
1) Identifying fake profiles.

2) Classifying a species.

3) Predicting what sport someone plays.

## Classification Tree - Overview

* Objective is to **infer class labels** from previously unseen data.

* Algorithmically, it is a solution that is both **recursive**(*divide & conquer*) and **greedy**(*favors optimization*). 

* Is a sequence of if-else questions about features, essentially playing "20 Questions".

* Captures **non-linear relationships** between features and labels.

* Is **non-parametric**, based on observed data and does not assume a normal distribution.

* Doesn't require **pre-processing**/feature scaling
_______________________________________________________________________________________________________________________________________

## Classification Model - Approach

Let's briefly set a mental framework for approaching the creation of a classification model. 

Like all things data, we'll begin with a **dataset**. 

Contained in the dataset(hopefully!) is organized tabular data with **records/observations, features**, and a **target**.

Next, we'll need to use these to create **training** and **test** sets. 

We can then use the **training** dataset with a **learning algorithm** (in our case, the **scikit-learn Decision Tree Classifier** module) to create a **model** via **induction**, which is then applied to make predictions on the test set of data through **deduction**. 

Here's a general schematic view of the concept. 

<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/workflow.png" class="inline"/><br>    
`Source: https://www-users.cs.umn.edu/~kumar001/dmbook/dmslides/chap4_basic_classification.pdf`
_______________________________________________________________________________________________________________________________________
## Model Concerns
As powerful as the technique can be, it needs a strong foundational awareness and human-level quality controls.

Here are some points to consider as you prepare for your **ML** task:

**A)** Are you selecting the **right problem** to test?

**B)** Are you capable of supplying **sufficient data**?

**C)** Are you providing your model **clean data**?

**D)** Are you able to prevent algorithmic **biases and confounding factors**?
_______________________________________________________________________________________________________________________________________
## Performance Metrics

It should be noted at the onset, that simple decision trees are **highly prone** to **overfitting**, leading to models which are **difficult to generalize**. One method of mitigating this potential risk is to engage in **pruning** of the tree, i.e., removing parts of the tree which confer too much **specificity** to the model. We will discuss methods of pruning shortly. 

*A cautious interpretation of seemingly powerful results is strongly encouraged.

**Goal**: Achieve the highest possible **accuracy**, while retaining the lowest **error rate**.
_______________________________________________________________________________________________________________________________________
### Classification & General Metrics    

**Accuracy Score**  
* `accuracy_score(y_test, y_pred)`    
The accuracy score is calculated through the ratio of the correctly predicted data points divided by all predicted data points.  

**Score**  
* `score(features, target)`    
Mean accuracy on the given test data and labels 

**Classification Report**    
* `classification_report(y_test, y_pred)`        
The confusion matrix produces  `precision  recall  f1-score   support` results.  

**Feature Importance**    
* `model.feature_importances_`    
Identifies the features with the most weight in the model.

**Confusion Matrix**  
* `confusion_matrix(y_test, y_pred)`    
Summarizes error rate in terms of true/false positives/negatives.  

While the rest of the tests outlined above return simple numbers to interpret, the confusion matrix needs a primer on output.

<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/matrix.png" class="inline"/><br>    
`Source: https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-13-S4-S2`

Calling `confusion_matrix()` will yield a result in the form:    
`[TP,FP]`   
`[FN,TN]`  
_______________________________________________________________________________________________________________________________________
### Regression Metrics
Even though we'll be sticking to using the Classification portion of the CART suite, let's briefly review the methods of regression performance analysis. 

**R-Squared**    
* `r2_score(y_true, y_pred)`
Goodness of fit, how well model fits regression line.

**Mean Squared Error**  
* `mean_squared_error(y_test, y_pred)`    
Computed average squared difference between the estimated values, and what is being estimated.  

**Mean Absolute Error**  
* `mean_absolute_error(y_true, y_pred)`    
The mean absolute error reflects the magnitude of difference between the prediction and actual.  

_______________________________________________________________________________________________________________________________________
## Tree Data Structure - Fundamentals
### Reference Image
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/tree_anatomy.jpeg" class="inline"/><br>
`Source: Data Structure Tree Diagram - artbattlesu.com`

Unlike common **linear data structures**, like **lists** and `arrays`, a **Tree** is a **non-linear**, **hierarchical** method of storing/modeling data. Visually, you can picture an evolutionary tree, a **document object model (DOM)** from HTML, or even a company's organizational flowchart. 

In contrast to a biological tree originating of kingdom plantae, the data structure tree has a simple anatomy:

A **tree** consists of **nodes** and **edges**. 

There are specialized node types classified by unique names which represent their place on the hierarchy of the structure. 

The **root** node, is a single point of origin for the rest of the tree. **Branches** can extend from nodes and link to other nodes, with the **link** referred to as an **edge**. 

The node which accepts a link/edge, is said to be a **child** node, and the originating node is the **parent**. 

A single node may have one, two, or no children. If a node has no children, but does have a parent, it is called a **leaf**. Some will also refer to **internal** nodes, which have one parent and two children. 

Finally, **sibling** nodes are nodes which share a parent. 

Beyond the core anatomy, the tree has unique metrics to be explored: **depth** and **height**. 

**Depth** refers to the spatial attributes of an individual node in relation to the root, meaning, how many links/edges are between the specific node and the root node. You could also think of it as the position of the node from `root:0` to `leaf:m` depth. 

The **height**, refers to the number of edges in the longest possible path of the tree, similar to finding the longest carbon chain back in organic chemistry to determine the IUPAC name of the compound.

## Decision Tree
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/Decision_tree_anatomy.jpg" class="inline"/><br>    
`Source: Machine Learning 10601 Recitation 8 Oct 21, 2009 Oznur Tastan`

A **decision tree**, allows us to run a series of **if/elif/else** tests/questions on a data point, record, or observation with many attributes to be tested. Each node of this tree, would represent some condition of an attribute to test, and the edges/links are the results of this test constrained to some kind of binary decision. An observation travels through each stage, being assayed and partitioned, to reach a leaf node. The leaf contains the final proposed classification. 

For example, if we had a dataset with rich features about a human, we could ask many questions about that person and their behavior based on gender(M/F/O), weight(Above/Below a value), height(Above/Below a value), activities(Sets of choices) to make a class-level prediction. 
_______________________________________________________________________________________________________________________________________
## Classification Plot - Simple

|              |               |
| ------------ | ------------- |
| **Decision Region** | Space where instances become assigned to particular class, blue or red, in the plotting space in the diagram.|     
| **Decision Boundary** | Point of transition from one decision region to another, aka one class/label to another, the diagonal black line. |

<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/Boundary.jpg" class="inline"/><br>  
`Source: Vipin Kumar CSci 8980 Fall 2002 13 Oblique Decision Trees`

_______________________________________________________________________________________________________________________________________
## Feature Selection

Identification of which features/columns with the highest weights in predictive power. We can manually view the importance of our features with `model.feature_importances_` as noted earlier.

In our case, the CART algorithm will do the feature selection for us through the **Gini Index** or **Entropy** which measure how pure your data is partitioned through it's journey to a final leaf node classification.
_______________________________________________________________________________________________________________________________________
## What is Purity?    
**Measures:**    
Effectiveness of class separation   

**Types:**    
* Entropy - Measure of randomness, distribution of class occurrences
* Gini - Measure of misclassification   

**Goal:**        
Purely separate leaf class outcomes with low rates of misclassification.   

## Gini & Entropy Analysis Tools
We can quickly test the accuracy when swapping both criterion and depth with the following methods.

## Gini Accuracy Test, Multiple `Max_Depths`
```Python3
def gini_depth_test(depth, X_train, y_train, y_test, X_test):
    
    """     
    :param depth: max depth
       
    :params X_train, y_train: X and Y training sets

    :params  y_test, X_test: X and Y testing sets 

    :return: df, collection of results w/ cols depth and score     
    
    """ 
 
    score_list = []
    
    depth_count = []     
    
    for cur_depth in range(1, depth):       
    
        dtc_gini = DecisionTreeClassifier(max_depth=cur_depth, criterion='gini',
                                                         random_state=cur_depth)       

        dtc_gini.fit(X_train,y_train)       

        y_pred_gini = dtc_gini.predict(X_test)       

        gini_score = accuracy_score(y_test, y_pred_gini)       

        score_list.append(round(gini_score, 3))       

        depth_count.append(cur_depth) 

    results = {'Depth': depth_count, 'Accuracy': score_list}     

    df = pd.DataFrame.from_dict(results)     

    return df
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/depths.png" class="inline"/><br>  
_______________________________________________________________________________________________________________________________________
## Entropy Accuracy Test, Multiple `Max_Depths`    
```Python3
def entropy_depth_test(depth, X_train, y_train, y_test, X_test):
    
    """     

    :param depth: max depth     
     
    :params X_train, y_train: X and Y training sets

    :params  y_test, X_test: X and Y testing sets 

    :return: df, collection of results w/ cols depth and score     
    
    """ 
 
    score_list = []
    
    depth_count = []     
    
    for cur_depth in range(1, depth):       
    
        dtc_entropy = DecisionTreeClassifier(max_depth=cur_depth, criterion='entropy',
                                                         random_state=cur_depth)       

        dtc_entropy.fit(X_train,y_train)       

        y_pred_entropy = dtc_entropy.predict(X_test)       

        entropy_score = accuracy_score(y_test, y_pred_entropy)       

        score_list.append(round(entropy_score, 3))       

        depth_count.append(cur_depth) 

    results = {'Depth': depth_count, 'Accuracy': score_list}     

    df = pd.DataFrame.from_dict(results)     

    return df
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/depth_2.png" class="inline"/><br>  
_______________________________________________________________________________________________________________________________________
## Pruning
As we discussed earlier, decision trees are prone to overfitting. 

**Pruning** is one way to mitigate the influence. As each node is a test, and each branch is a result of this test, we can **prune** unproductive branches which contribute to **overfitting**. By removing them, we can further **generalize the model**.

## Pre-Pruning
One strategy for pruning is known as **pre-pruning**. This method relies on ending the series of tests early, stopping the partitioning process. When stopped, what was previously a non-leaf node, becomes the leaf node and a class is declared. We can also utilize the validation set to check for overfitting. 

## Post-Pruning
**Post-Pruning** is a different approach. Where **pre-pruning** occurs during creation of the model, **post-pruning** begins after the process is complete through the **removal of branches**. Sets of node removals are tested throughout the branches, to examine the effect on error-rates. If removing particular nodes increases the error-rate, pruning does not occur at those positions. The final tree contains a version of the tree with the **lowest expected error-rate**. In this process, we end the tree when performance on the validation set begins losing performance. 
_______________________________________________________________________________________________________________________________________
## PART II - THE WORKFLOW     
## Steps to Build and Run    

1 **Imports**

2 **Load Data**

3 **Test and Train Data**

4 **Instantiate a Decision Tree Classifier**

5 **Fit data**

6 **Predict**

7 **Check Performance Metrics**
_______________________________________________________________________________________________________________________________________
## 1-Import Modules/Libraries 
```Python3
from sklearn.tree import DecisionTreeClassifier 
 
from sklearn.model_selection import train_test_split 
 
from sklearn.metrics import accuracy_score 
 
from sklearn.metrics import confusion_matrix , classification_report 
```

## 2-Load Data

First, we're going to want to load a dataset, and create two sets, X and y, which represent our features and our desired label.

```Python3
X, y = dataset.data, dataset.target 
 
features = iris.feature_names
```

## 3-Split Dataset into Test and Train sets
Now, we can partition our data into test and train sets, and the typical balance is usually 20/80 or 30/70 test vs train percentages. 

The results of the split will be stored in `X_train`, `X_test`, `y_train`, and `y_test`. 

```Python3
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
```

## 4-Instantiate a DecisionTreeClassifier
We can instantiate our **DecisionTreeClassifier** object with a `max_depth`, and a `random_state`.

`random_state` allows us a way of **ensuring reproducibility**, while `max_depth` is a **hyper-parameter**, which allows us to **control complexity** of our tree, but must be used with a cautious awareness. If `max_depth` is set too high, we risk **over-fitting** the data, while if it's too low, we will be **underfitting**. 

```Python3
dt = DecisionTreeClassifier(max_depth=desired_depth, random_state=1)
```

## 5-Fit The Model
We fit our our model by utilizing `.fit()` and feeding it parameters `X_train` and `y_train` which we created previously.

```Python3
dt.fit(X_train, y_train)
```

## 6-Predict Test Set Labels
We can now test our model by applying it to our `X_test` variable, and we'll store this as `y_pred`. 

```Python3
y_pred = dt.predict(X_test)
```

## 7-Check Performance Metrics

We want to check the accuracy of our model, so let's run through some performance metrics by turning them into simple, reusable method with tidy outputs.

**Accuracy Score**
```Python3
def AccuracyCheck(model, X_test, y_pred):

    """ 
  
    :param model: trained model 
    
    :param X_test: Feature test data 

    :param y_pred: Class predict 

    :return: None, prints accuracy score 

    """ 

    acc = accuracy_score(y_test, y_pred)

    print('Default Accuracy: {}'.format(round(acc), 3))
```

**Classification Report** 
```Python3
def ClassificationReport(y_test, y_pred):

    """ 
    :param y_test: Target test 

    :param y_pred: Class predict
 
    :return: None, prints classification report

    """ 

    cr = classification_report(y_test, y_pred)

    print('Classification Report: \n{}'.format(cr))
```

**Confusion Matrix**
```Python3
def ConfusionMatx(y_test, y_pred):
   
    """ 
    :param y_test: Target test  
     
    :param y_pred: Class predict 
    
    :return: None, prints confusion matrix 
    
    """ 
    print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, y_pred)) 
```

**Score**
```Python3
def DTCScore(X, y, dtc):
   
    """ 
    :param X: Features 

    :param y: Target 
    
    :param dtc: instance of DTC 
    
    :return: None, prints score 
    
    """ 
    score = dtc.score(X, y, sample_weight=None) 
    
    print('Score: {}'.format(round(score))) 
```    

**Feature Importance**
```Python3
def feature_finder(df, model):

    """

    Calculates and prints feature importance

    :args: df - dataframe of dataset

           model - fitted model

    :return none:

    """

    features = dict(zip(df.columns, model.feature_importances_))

    for feature, score in features.items():

        print(feature, round(score,3))
```
**Summary Report**    
Thanks to the power of Python, we can run all of the tests in one go via scripting:
```Python3
def MetricReport(X, y, y_test, y_pred, dtc):""" 
    
    :param X: Features 
        
    :param y: Target 
    
    :param y_test: feature test set 
    
    :param y_pred: Class predict 
    
    :param dtc: instance of DTC 
     
    :return: None, prints collection of metric reports 
    
    """ 
    print("Metric Summaries") 
    
    print("-"*16)
    
    ConfusionMatx(y_test, y_pred) 
     
    DTCScore(X, y, dtc) 
    
    classification_report(y_test, y_pred) 
    
    print("-" * 16) 
```
_______________________________________________________________________________________________________________________________________
# PART III - CASE STUDY
## Hepatitis Survival

We'll follow the procedures above, with a few twists. We're going to add a way to **visualize** our decision tree **graph**, using a real dataset with the tools and approaches outlined.

## Imports
```Python3
import pandas as pd

import graphviz

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
```

## Dataset
https://archive.ics.uci.edu/ml/datasets/Hepatitis

```Python3
path = '...hepatitis.csv'

col_names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise',
             'Anorexia', 'Liver_Big', 'Liver_Firm', 'Spleen_Palp', 'Spiders', 
             'Ascites', 'Varices', 'Bilirubin', 'Alk_Phosph', 'SGOT', 'Albumin',
                                                          'Protime','Histology' ]

csv = pd.read_csv(path, na_values=["?"], names=col_names)

df = pd.DataFrame(csv)
```

## Survey The Data
```Python3
def minorEDA(df):

    """

    Generates a preliminary EDA Analysis of our file

    args: df - DataFrame of our excel file

    returns: None

    """

    lineBreak = '------------------'

    #Check Shape

    print(lineBreak*3)

    print("Shape:")

    print(df.shape)

    print(lineBreak*3)

    #Check Feature Names

    print("Column Names")

    print(df.columns)

    print(lineBreak*3)

    #Check types, missing, memory

    print("Data Types, Missing Data, Memory")

    print(df.info())

    print(lineBreak*3)
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/EDA_1_Hep.png" class="inline"/><br>

## Check Integrity
```Python3
def check_integrity(input_df):

    """  

    Check if values missing, generate list of cols with NaN indices
     
    Args: input_df - Dataframe

    Returns: List containing column names containing missing data

    """

    if df.isnull().values.any():

        print("\nDetected Missing Data\nAffected Columns:")

        affected_cols = [col for col in input_df.columns if input_df[col].isnull().any()]
        
        affected_rows = df.isnull().sum()

        missing_list = []

        for each_col in affected_cols:

            missing_list.append(each_col)

        print(missing_list)

        print("\nCounts")

        print(affected_rows)

        print("\n")

        return missing_list

    else:

        pass

    print("\nNo Missing Data Was Detected.")
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/EDA_2_Hep.png" class="inline"/><br>

Unfortunately this set contains missing data points, if we don't clean them up, we'll recieve an error. To do this, `df.dropna(inplace=True)` in the `__main__` portion of our program will allow us to continue, but our model will ultimately be weaker due to missing inputs.

## Set and Retrieve Label, Features
```Python3
def set_target(dataframe, target):"""

    :param dataframe: Full dataset

    :param target: Name of classification column

    :return x: Predictors dataset

    :return y: Classification dataset

    """
    x = dataframe.drop(target, axis=1)

    y = dataframe[target]

    return x, y
 ```
 
 ## Decision Tree
 ```Python3
def DecisionTree():

   # Build Decision Tree Classifier

   dtc = DecisionTreeClassifier(max_depth=6, random_state=2)

   return dtc
```

## Test, Train
```Python3
def TestTrain(X, y):

    """

    :param X: Predictors

    :param y: Classification

    :return: X & Y test/train data


    """# Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                 stratify=y, random_state=1)
    return X_train, X_test, y_train, y_test
```

## Fit
```Python3
def FitData(DTC, X_train, y_train):

    # Fit training data

    return DTC.fit(X_train, y_train)
```

## Predict
```Python3
def Predict(dtc, test_x):

    y_pred = dtc.predict(test_x)

    return y_pred
```

## Check Accuracy, Metrics, Generate Report
```Python3
def AccuracyCheck(model, X_test, y_pred):

    acc = accuracy_score(y_test, y_pred)

    print('Default Accuracy: {}'.format(round(acc), 3))
    
    
def ConfusionMatx(y_test, y_pred):

    print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))


def DTCScore(X, y, dtc):

    score = dtc.score(X, y, sample_weight=None)

    print('Score: {}'.format(round(score)))


def feature_finder(df, model):

    """

    Calculates and prints feature importance

    :args: df - dataframe of dataset
     
    model - fitted model

    :return none:

    """

    features = dict(zip(df.columns, model.feature_importances_))

    for feature, score in features.items():

        print(feature, round(score,3))

        
        
def MetricReport(X, y, y_test, y_pred, dtc):

    print("Metric Summaries")

    print("-"*16)

    ConfusionMatx(y_test, y_pred)

    DTCScore(X, y, dtc)

    classification_report(y_test, y_pred)

    print("-" * 16)
```
## Feature Weights
```Python3
def feature_finder(df, model):

    """

    Calculates and prints feature importance

    :args: df - dataframe of dataset
     
    model - fitted model

    :return none:

    """

    features = dict(zip(df.columns, model.feature_importances_))

    for feature, score in features.items():

        print(feature, round(score,3))
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/features.png" class="inline"/><br>
**Albumin**, **bilirubin** and **sex** appear to have the greatest influence on survival classification. 

## Plot Features    
Let's go ahead and create a small method to generate a plot of features so we can visually assay the distributions.
```Python3
def plot_features(feature_dict):

    '''

    :param feature_dict: Dictionary of feature weights in k,v pairs

    :retur: None, Displays bar plot of features and model weights

    '''

    feature_dict = dict((k, v) for k, v in feature_dict.items() if v >= 0.01)

    names = list(feature_dict.keys())

    values = list(feature_dict.values())

    values = values

    plt.bar(names, values)

    plt.xlabel('Categories')

    plt.ylabel('Percentage\n(%)')

    plt.title('Feature Weight')

    plt.show()
 ```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/myplot.png" class="inline"/><br>  

## Metric Report
```Python3
        
def MetricReport(X, y, y_test, y_pred, dtc):

    print("Metric Summaries")

    print("-"*16)

    ConfusionMatx(y_test, y_pred)

    DTCScore(X, y, dtc)

    classification_report(y_test, y_pred)

    print("-" * 16)
```
<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/metric_report.png" class="inline"/><br>

Accuracy and Score both indicate over-fitting is occurring. The confusion matrix yielded 3 true positives, 16 true negatives, 1 false positive and 4 false negatives. The classification report gives more realistic results, averaging 86% model precision. It's possible that survival can be predicted by the indicators of liver damage(albumin/bilirubin), but we should always take results with a dose of healthy skepticism

## Visualize Tree Graph
```Python3
def tree_viz(dtc, classes, col_names):

""" 

:param dtc: Decision Tree Instance 

:param classes: list of categorical outcomes, Ex. ['Non-Survival', 'Survival']

:param col_names: list of feature names 

:return: None, Plots a Decision Tree Graph 

""" 

class_n = classes 

dot = tree.export_graphviz(dtc, out_file=None, feature_names=col_names, 

class_names=classes, filled=True, rounded=True) 

graph = graphviz.Source(dot)

graph.format = 'png' 

graph.render('Hep', view=True) 
```

<img src="https://github.com/ajh1143/ajh1143.github.io/blob/master/Images/DTC/graph_viz.png" class="inline"/><br>
_______________________________________________________________________________________________________________________________________
## Run
```Python3
if __name__ == '__main__': 

path = 'C:\\Users\\ajh20\\Desktop\\hepatitis.csv' 

col_names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 

'Malaise','Anorexia', 'Liver', 'Spleen_Palp', 'Spiders', 'Ascites', 

'Varices', 'Bilirubin', 'Alk_Phosph', 'SGOT', 'Alb 'Histology' ]

 classes = ['Non-Survival', 'Survival'] 

csv = pd.read_csv(path, na_values=["?"], names=col_names) 

df = pd.DataFrame(csv) 

minorEDA(df) 

check_integrity(df)
 
df.dropna(inplace=True) 

X, y = set_target(df, 'Class') 

dtc = DecisionTree() 

X_train, X_test, y_train, y_test = TestTrain(X, y) 

model_test = FitData(dtc, X_train, y_train) 

feature_finder(df, model_test) 

y_pred = Predict(dtc, X_test) 

tree_viz(dtc, classes, col_names) 

MetricReport(X, y, y_test, y_pred, dtc) 
```
_______________________________________________________________________________________________________________________________________
## Review:    
We've learned what CART algorithms are, when to use the, and some of their quirks. We also developed an understanding of tree data structures, performance metrics for classification and regression, built some great reusable tools and applied our knowledge through a study on hepatitis sutvival.
_______________________________________________________________________________________________________________________________________
## Full Code
```Python3
import pandas as pd 
 
import graphviz 
 
from sklearn import tree 
 
from sklearn.tree import DecisionTreeClassifier 
 
from sklearn.model_selection import train_test_split 
 
from sklearn.metrics import accuracy_score 
 

 
def set_target(dataframe, target): 
 
    """ 
 
    :param dataframe: Full dataset 
 
    :param target: Name of classification column 
 
    :return x: Predictors dataset 
 
    :return y: Classification dataset 
 
    """ 
 
    x = dataframe.drop(target, axis=1) 
 
    y = dataframe[target] 
 
    return x, y 
 
 
 
def TestTrain(X, y): 
 
    """ 
 
    :param X: Predictors 
 
    :param y: Classification 
 
    :return: X & Y test/train data 
 
    """ 
 
    # Split into training and test set 
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      stratify=y, random_state=1)
 
    return X_train, X_test, y_train, y_test 
 
 

def DecisionTree(): 
 
    # Build Decision Tree Classifier 
 
    dtc = DecisionTreeClassifier(max_depth=6, random_state=2) 
 
    return dtc 


 
def FitData(DTC, X_train, y_train):
    
    """

    :param DTC: Decision Tree Instance
   
    :param X_train: Feature training data 
 
    :param y_train: Classification training data 
 
    :return: fitted model 
 
    """ 

    return DTC.fit(X_train, y_train)

 

def Predict(dtc, test_x):""" 
 
    :param dtc: Decision Tree Instance 
 
    :param test_x: Test Feature set 
 
    :return: y_pred 
 
    """ 

    y_pred = dtc.predict(test_x)

    return y_pred



def AccuracyCheck(model, y_test, y_pred): 
 
    """ 
 
    :param model: trained model 
 
    :param X_test: Feature test data          

    :param y_pred: Class predict 
 
    :return: None, prints accuracy score 
 
    """ 
 

    acc = accuracy_score(y_test, y_pred)  
   
    print('Default Accuracy: {}'.format(round(acc), 3)) 



def ConfusionMatx(y_test, y_pred):"""     
    :param y_test: Target test     

    :param y_pred: Class predict 
 
    :return: None, prints accuracy score    

     """ 
 
    print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, y_pred))) 
 
 
 
 
 
def MeanAbsErr(y_test, y_pred):

    """    

    :param y_test: Target test      

    :param y_pred: Class predict 
 
    :return: None, prints MAE   

    """ 
 
    mean_err = metrics.mean_absolute_error(y_test, y_pred) 
 
    print('Mean Absolute Error: {}'.format(round(mean_err), 3))  



def MeanSqErr(y_test, y_pred):

    """

    :param y_pred: Class predict 
 
    :return: None, prints MSE    
   
    """ 
 
    SqErr = metrics.mean_squared_error(y_true, y_pred) 
 
    print('Mean Squared Error: {}'.format(round(SqErr), 3))



def DTCScore(X, y, dtc):

    """    

    :param X: Features  
   
    :param y: Target  
 
    :param dtc: instance of DTC 
  
    :return: None, prints score   
    
    """ 
 
    score = dtc.score(X, y, sample_weight=None) 
 
    print('Score: {}'.format(round(score))) 
 
 
 
 
 
def feature_finder(df, model): 
 
    """ 
 
    Calculates and prints feature importance 
 
    :args: df - dataframe of dataset 
 
           model - fitted model 
 
    :return none: 
 
    """ 
 
    features = dict(zip(df.columns, model.feature_importances_)) 
 
    print(features) 
 
 
 
 
 
def gini_depth_test(depth, X_train, y_train, y_test, X_test):
   
     """     
    :param depth: max depth       
  
    :params X_train, y_train: X and Y training sets  

    :params  y_test, X_test: X and Y testing sets 
 
    :return: df, collection of results w/ cols depth and score    

     """ 
 
    score_list = []     
    
    depth_count = []     
    
    for cur_depth in range(1, depth):       

        dtc_gini = DecisionTreeClassifier(max_depth=cur_depth, criterion='gini',
                               
                                                         random_state=cur_depth)
                                                                 
        dtc_gini.fit(X_train,y_train)       

        y_pred_gini = dtc_gini.predict(X_test)       

        gini_score = accuracy_score(y_test, y_pred_gini)      
 
        score_list.append(round(gini_score, 3))   
    
        depth_count.append(cur_depth)

    results = {'Depth': depth_count, 'Accuracy': score_list}     

    df = pd.DataFrame.from_dict(results)    
 
    return df 
 

 
def entropy_depth_test(depth, X_train, y_train, y_test, X_test):
       
    """     
    :param depth: max depth 
 
    :params X_train, y_train: X and Y training sets           

    :params  y_test, X_test: X and Y testing sets 
 
    :return: df, collection of results w/ cols depth and score 
 
    """ 
 
    score_list = []     

    depth_count = []   

    for cur_depth in range(1, depth):     


        dtc_entropy = DecisionTreeClassifier(max_depth=cur_depth, criterion='entropy',                                                                

                                                             random_state=cur_depth)       

        dtc_entropy.fit(X_train,y_train)      

        y_pred_entropy = dtc_entropy.predict(X_test)      

        entropy_score = accuracy_score(y_test, y_pred_entropy)     

        score_list.append(round(gini_score, 3))     

        depth_count.append(cur_depth) 

    results = {'Depth': depth_count, 'Accuracy': score_list}    
   
    df = pd.DataFrame.from_dict(results)     
    
    return df 

```
