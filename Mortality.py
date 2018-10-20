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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    return X_train, X_test, y_train, y_test

def DecisionTree():
    # Build Decision Tree Classifier
    dtc = DecisionTreeClassifier(max_depth=6, random_state=2)
    return dtc

def FitData(DTC, X_train, y_train):
    # Fit training data
    return DTC.fit(X_train, y_train)

def Predict(dtc, test_x):
    y_pred = dtc.predict(test_x)
    return y_pred

def AccuracyCheck(model, X_test, y_pred):
    #Cneck Accuracy Score
    acc = accuracy_score(y_test, y_pred)
    print('Default Accuracy: {}'.format(round(acc), 3))

    def ConfusionMatx(y_test, y_pred):
    print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

def MeanAbsErr(y_test, y_pred):
    mean_err = metrics.mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {}'.format(round(mean_err), 3))

def MeanSqErr(y_test, y_pred):
    SqErr = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: {}'.format(round(SqErr), 3))

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
    print(features)

def MetricReport(df, X, y, y_test, y_pred, dtc, model):
    print("Metric Summaries")
    print("-"*16)
    feature_finder(df, model)
    ConfusionMatx(y_test, y_pred)
    MeanAbsErr(y_test, y_pred)
    MeanSqErr(y_test, y_pred)
    DTCScore(X, y, dtc)
    print("-" * 16)

def tree_viz(dtc, df, col_names):
    class_n = ["Non-Survival", "Survival"]
    dot = tree.export_graphviz(dtc, out_file=None, feature_names=col_names, class_names=class_n, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot)
    graph.format = 'png'
    graph.render('iris', view=True)


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

def check_integrity(input_df):
    """  Check if values missing, generate list of cols with NaN indices
     Args:
             input_df - Dataframe
     Returns:
             List containing column names containing missing data
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

path = 'C:\\Users\\ajh20\\Desktop\\hepatitis.csv'
col_names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise','Anorexia', 'Liver_Big', 'Liver_Firm',
             'Spleen_Palp', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk_Phosph', 'SGOT', 'Albumin', 'Protime',
             'Histology' ]
csv = pd.read_csv(path, na_values=["?"], names=col_names)
df = pd.DataFrame(csv)
minorEDA(df)
check_integrity(df)
df.dropna(inplace=True)
X, y = set_target(df, 'Class')
dtc = DecisionTree()
X_train, X_test, y_train, y_test = TestTrain(X, y)
model_test = FitData(dtc, X_train, y_train)
y_pred = Predict(dtc, X_test)
AccuracyCheck(model_test, X_test, y_pred)
tree_viz(dtc, df, col_names)
MetricReport(df, X, y, y_test, y_pred, dtc, model_test)
