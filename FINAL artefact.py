import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import askyesno
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
import multiprocessing.pool as mpool
import threading
import time
from playsound import playsound

def show_results():
    display_text.delete(1.0, tk.END)
    display_text.insert(tk.END, df_Results.to_string())


def show_info_start_text():
    label = dataLabel_entry.get()
    df = pd.read_csv(input_file_entry.get())
    percentile = (df.groupby(label)[label].count() / df[label].count()) * 100
    display_text.delete(1.0, tk.END)
    a = str(df.isnull().values.any())

    display_text.insert(tk.END, "Showing basic information on your dataset press button above to visualise data\n")
    display_text.insert(tk.END, "Null values in dataset?:")
    display_text.insert(tk.END, str(a))
    display_text.insert(tk.END, "\nBalanced dataset?\n:")
    display_text.insert(tk.END, str(is_balanced()))
    display_text.insert(tk.END, "\n\nHead\n")
    display_text.insert(tk.END, str(df.head()))
    display_text.insert(tk.END, "\nShape")
    display_text.insert(tk.END, str(df.shape))
    display_text.insert(tk.END,
                        "\nFor Larger Datasets it is advisable not to use SVM unless you are using undersampling as you will run into performance issues.")
    display_text.insert(tk.END, "\nDistribution of value types in label\n")
    display_text.insert(tk.END, str(df[label].value_counts()))
    display_text.insert(tk.END, "\nDistribution of value types in label by percentage\n")
    display_text.insert(tk.END, percentile)


def is_balanced():
    label = dataLabel_entry.get()
    df = pd.read_csv(input_file_entry.get())
    value_counts_in_dataset = list(df[label].value_counts())
    value_counts_in_dataset.sort()
    is_balanced = bool
    balance = (value_counts_in_dataset[1] / value_counts_in_dataset[0])
    if (balance < 3):
        is_balanced = True
    else:
        is_balanced = False
    return is_balanced


def create_histograms():
    label = dataLabel_entry.get()
    df = pd.read_csv(input_file_entry.get())
    X = df.drop(['Class'], axis=1)
    cols = list(X.columns.values)
    normal_records = df.Class == 0
    abnormal_records = df.Class == 1
    plt.figure(figsize=(20, 60))
    for n, col in enumerate(cols):
        plt.subplot(10, 3, n + 1)
        sns.distplot(X[col][normal_records], color='green')
        sns.distplot(X[col][abnormal_records], color='red')
        plt.title(col, fontsize=17)
    plt.show()


def show_info_start_visual():
    label = dataLabel_entry.get()
    df = pd.read_csv(input_file_entry.get())
    plt.figure(figsize=(27, 17))
    plt.title("correlation HeatMap", fontsize=14)
    sns.heatmap(df.corr(), cmap="binary", annot=True)
    plt.figure(figsize=(15, 20))
    ((df.groupby(label)[label].count() / df[label].count()) * 100).plot.pie()
    plt.show()


def show_plots():
    plt.show()


def choose_label(labels):
    popup = tk.Toplevel()
    popup.title('Choose Label')
    dataLabel_label = tk.Label(popup, text="Choose label to be used in the algorithm:")
    dataLabel_label.pack()
    dataLabel_variable = tk.StringVar(popup)
    dataLabel_dropdown = tk.OptionMenu(popup, dataLabel_variable, *labels)
    dataLabel_dropdown.pack()
    tk.Button(popup, text='Close', command=popup.destroy).pack()
    popup.wait_window()
    return dataLabel_variable


def choose_IDLabel():
    popup = tk.Toplevel()
    popup.title('Choose Label')
    df = pd.read_csv(input_file_entry.get())
    labels = list(df.columns.values)
    IDLabel_label = tk.Label(popup, text="Choose ID label:")
    IDLabel_label.pack()
    IDLabel_variable = tk.StringVar(popup)
    IDLabel_dropdown = tk.OptionMenu(popup, IDLabel_variable, *labels)
    IDLabel_dropdown.pack()
    tk.Button(popup, text='Close', command=popup.destroy).pack()
    popup.wait_window()
    return IDLabel_variable.get()


def browse_file():
    file_path = filedialog.askopenfilename()
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)
    df = pd.read_csv(input_file_entry.get())
    column_headers = list(df.columns.values)
    label = choose_label(column_headers)
    dataLabel_entry.insert(0, label.get())
    show_info_start_text()


import warnings

warnings.filterwarnings("ignore")
def Plot_confusion_matrix(y_test, pred_test):
    plt.figure(figsize=(25, 25))
    cm = confusion_matrix(y_test, pred_test)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap="OrRd")
    categoryNames = ['Normal', 'Abnormal']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ticks = np.arange(len(categoryNames))
    plt.xticks(ticks, categoryNames, rotation=45)
    plt.yticks(ticks, categoryNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]), fontsize=12)


def Methodology_Name(balancing, crossvalidation):
    Methodology = ""
    if balancing == "None":
        if crossvalidation == "None":
            Methodology = "None"
        else:
            Methodology = crossvalidation + ("With no sampling")
    else:
        Methodology += balancing
        if crossvalidation == "None":
            pass
        else:
            Methodology += (" With " + crossvalidation)
    return Methodology


def identify_potential_outliers(model, aX, yy):
    global df_Results
    df = pd.read_csv(input_file_entry.get())
    label = dataLabel_entry.get()
    y = df[label]
    X = df.drop([label], axis=1)
    outlier_prob = model.predict_proba(X)[:, 1]
    iDlabel = choose_IDLabel()
    dataLabel = dataLabel_entry.get()
    results = pd.DataFrame({iDlabel: X[iDlabel], dataLabel: y, 'Outlier Probability': outlier_prob})
    results.to_csv('results.csv', index=False)
    potential_outliers = results[results['Outlier Probability'] > 0.5]
    potential_outliers.to_csv('potential_outliers.csv', index=False)


def identify_potential_outliers2(model, X, y):
    # global df_Results
    # df = pd.read_csv(input_file_entry.get())
    # label = dataLabel_entry.get()
    # y = df[label]
    # X = df.drop([label], axis=1)
    score = model.score_samples(X)
    outlier_prob = np.exp(score) / (np.exp(score) + 1)
    iDlabel = choose_IDLabel()
    dataLabel = dataLabel_entry.get()
    results = pd.DataFrame({iDlabel: X[iDlabel], dataLabel: y, 'Outlier Probability': outlier_prob})
    results.to_csv('results.csv', index=False)
    potential_outliers = results[results['Outlier Probability'] < np.percentile(outlier_prob, 2.5)]
    potential_outliers.to_csv('potential_outliers.csv', index=False)

def EvaluateLogisticModelsL1( Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    model = LogisticRegression(penalty='l1',solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_probs = model.predict_proba(X_test)[:, 1]
    Accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
    roc_value = roc_auc_score(y_test, y_pred_probs)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'Logistic Regression L1', 'Accuracy': Accuracy,
         'roc_value': roc_value, 'threshold': threshold}, index=[0]), ignore_index=True)
    results = {}
    results['name'] = Methodology + " Logisitic Regression L1"
    results['accuracy'] = Accuracy
    results['classification_report'] = classification_report(y_test, y_pred)
    results['roc'] = roc_value
    global reports
    reports.append(results)
    if Output == True:
        print("Accuarcy of Logistic model: {0}".format(Accuracy))
        Plot_confusion_matrix(y_test, y_pred)
        print("classification Report")
        print(classification_report(y_test, y_pred))


        print("roc_value: {0}".format(roc_value))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)


        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))

        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)



        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers(model, X_test, y_test)
        return results
def EvaluateLogisticModelsL2(Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    model = LogisticRegression(penalty='l2',solver='newton-cg')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    Accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
    roc_value = roc_auc_score(y_test, y_pred_probs)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'Logistic Regression L2', 'Accuracy': Accuracy,
         'roc_value': roc_value, 'threshold': threshold}, index=[0]), ignore_index=True)
    results = {}
    results['name'] = Methodology + " Logisitic Regression L2"
    results['accuracy'] = Accuracy
    results['classification_report'] = classification_report(y_test, y_pred)
    results['roc'] = roc_value
    global reports
    reports.append(results)
    if Output == True:
        print("Accuarcy of Logistic model: {0}".format(Accuracy))
        Plot_confusion_matrix(y_test, y_pred)
        print("classification Report")
        print(classification_report(y_test, y_pred))
        print("roc_value: {0}".format(roc_value))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)

        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))

        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)

        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers(model, X_test, y_test)
        return results

def EvaluateKNNModels(Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=16)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    knn_probs = model.predict_proba(X_test)[:, 1]
    knn_roc_value = roc_auc_score(y_test, knn_probs)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, knn_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = model.predict(X_test)
    KNN_Accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)

    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'KNN', 'Accuracy': score, 'roc_value': knn_roc_value,
         'threshold': threshold}, index=[0]), ignore_index=True)
    results = {}
    results['name'] = Methodology + " KNN"
    results['accuracy'] = KNN_Accuracy
    results['classification_report'] = classification_report(y_test, y_pred)
    results['roc'] = knn_roc_value
    global reports
    reports.append(results)
    if Output:
        print("model score")
        print(score)

        y_pred = model.predict(X_test)

        Plot_confusion_matrix(y_test, y_pred)
        print("classification Report")
        print(classification_report(y_test, y_pred))


        print("KNN roc_value: {0}".format(knn_roc_value))

        print("KNN threshold: {0}".format(threshold))

        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))
        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)

        results['name'] = (Methodology, " KNN")
        results['accuracy'] = KNN_Accuracy
        results['classification_report'] = classification_report(y_test, y_pred)
        results['roc'] = knn_roc_value



        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers(model, X_test, y_test)
        return results


def EvaluateSVMModels(Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    print("Confusion ")

    clf = SVC(kernel='sigmoid', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    svm_probs = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    SVM_Score = accuracy_score(y_test, y_pred)
    roc_value = roc_auc_score(y_test, svm_probs)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, svm_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'SVM', 'Accuracy': SVM_Score, 'roc_value': roc_value,
         'threshold': threshold}, index=[0]), ignore_index=True)

    results = {}
    results['name'] = Methodology + " SVM"
    results['accuracy'] = SVM_Score
    results['classification_report'] = classification_report(y_test, y_pred)
    results['roc'] = roc_value
    global reports
    reports.append(results)
    if Output == True:
        print("accuracy_score : {0}".format(SVM_Score))

        Plot_confusion_matrix(y_test, y_pred)
        print("classification Report")
        print(classification_report(y_test, y_pred))

        # Calculate roc auc
        roc_value = roc_auc_score(y_test, svm_probs)

        print("SVM roc_value: {0}".format(roc_value))

        print("SVM threshold: {0}".format(threshold))
        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))
        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)
        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers(clf, X_test, y_test)
        return results

def EvaluateIsolationForestModels(Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    global reports
    # Create the model
    IF_model = IsolationForest(n_estimators=100,
                               contamination=0.2, random_state=np.random.RandomState(42), verbose=0)
    IF_model.fit(X_train)
    results = {}
    # Convert scores to binary predictions
    IF_score = IF_model.score_samples(X_test)
    IF_predictions = np.where(IF_score >= np.percentile(IF_score, 2.5), 0, 1)
    IF_probs = IF_model.score_samples(X_test)
    IF_probs = np.exp(IF_probs) / (np.exp(IF_probs) + 1)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, IF_predictions)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, IF_probs)
    roc_value = roc_auc_score(y_test, IF_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'Isolation Forest', 'Accuracy': accuracy, 'roc_value': roc_value,
         'threshold': threshold}, index=[0]), ignore_index=True)
    results = {}
    results['name'] = Methodology + "Isolation Forest"
    results['accuracy'] = accuracy
    results['classification_report'] = classification_report(y_test, IF_predictions)
    results['roc'] = roc_value
    reports.append(results)

    if Output == True:
        print('Model Accuracy: {0}'.format(accuracy))
        print("Confusion Matrix")
        Plot_confusion_matrix(y_test, IF_predictions)
        print("Classification Report")
        print(classification_report(y_test, IF_predictions))


        print("Isolation Forest roc_value: {0}".format(roc_value))
        print("Isolation Forest threshold: {0}".format(threshold))
        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))
        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)

        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers2(IF_model, X_test, y_test)
        return results



def EvaluateXGBoostModels(Methodology, X_train, y_train, X_test, y_test, Output):
    global df_Results
    XGBmodel = XGBClassifier(random_state=42)
    XGBmodel.fit(X_train, y_train)
    y_pred = XGBmodel.predict(X_test)
    df = pd.read_csv(input_file_entry.get())
    label = dataLabel_entry.get()
    y = df[label]
    X = df.drop([label], axis=1)
    Accuracy_Score = XGBmodel.score(X_test, y_test)

    XGB_probs = XGBmodel.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, XGB_probs)
    threshold = thresholds[np.argmax(tpr - fpr)]
    roc_value = roc_auc_score(y_test, XGB_probs)
    df_Results = df_Results.append(pd.DataFrame(
        {'Methodology': Methodology, 'Model': 'XGBoost', 'Accuracy': Accuracy_Score, 'roc_value': roc_value,
         'threshold': threshold}, index=[0]), ignore_index=True)
    results = {}
    results['name'] =  Methodology + " XGBoost"
    results['accuracy'] = Accuracy_Score
    results['classification_report'] = classification_report(y_test, y_pred)
    results['roc'] = roc_value
    global reports
    reports.append(results)
    if Output == True:
        print('Model Accuracy: {0}'.format(Accuracy_Score))

        print("Confusion Matrix")
        Plot_confusion_matrix(y_test, y_pred)
        print("classification Report")
        print(classification_report(y_test, y_pred))

        print("XGboost roc_value: {0}".format(roc_value))

        print(fpr, tpr, thresholds)

        print("XGBoost threshold: {0}".format(threshold))
        roc_auc = metrics.auc(fpr, tpr)
        print("ROC for the test dataset", '{:.1%}'.format(roc_auc))
        plt.figure(figsize=(10, 15))
        plt.plot(fpr, tpr, label="Test, auc=" + str(roc_auc))
        plt.legend(loc=4)
        createcsv = askyesno("Output results to CV",
                             "Would you like to output the the predicted outliers to a CSV file")
        if createcsv:
            identify_potential_outliers(XGBmodel, X_test, y_test)
        return results
root = tk.Tk()
root.title("Outlier Detection using Machine Learning Algorithms")
root.geometry("500x300")
browse_button = tk.Button(root, text="Select dataset(CSV)", command=browse_file)
browse_button.pack()
# Create an entry box for displaying the input data file path
input_file_label = tk.Label(root, text="Input Data File:")
input_file_label.pack()
input_file_entry = tk.Entry(root)
input_file_entry.pack()
dataLabel_label = tk.Label(root, text="Label to be used in ML algorithm")
dataLabel_label.pack()
dataLabel_entry = tk.Entry(root)
dataLabel_entry.pack()

visualise_button = tk.Button(root, text="Visualise", command=show_info_start_visual)
visualise_button.pack()
histogram_button = tk.Button(root, text="Create Histograms", command=create_histograms)
histogram_button.pack(side='right')
showplots_button = tk.Button(root, text="Show plots for recent model", command=show_plots)
showplots_button.pack(side='right')
# Create a dropdown menu for selecting the algorithm
algorithm_label = tk.Label(root, text="Algorithm:")
algorithm_label.pack()
algorithm_variable = tk.StringVar(root)
algorithm_dropdown = tk.OptionMenu(root, algorithm_variable, "SVM", "Isolation Forest", "KNN", "Logistic Regression L2","Logistic Regression L1",
                                   "XGBoost")
algorithm_dropdown.pack()
crossvalidation_variable = tk.StringVar(root)
crossvalidation_dropdown = tk.OptionMenu(root, crossvalidation_variable, "RepeatedKFold", "StratifiedKFold", "None")
crossvalidation_dropdown.pack(side="left")
balancing_variable = tk.StringVar(root)
balancing_dropdown = tk.OptionMenu(root, balancing_variable, "RandomOverSampler", "Smote", "RandomUnderSampler", "None")
balancing_dropdown.pack(side="left")
df_Results = pd.DataFrame(columns=['Methodology', 'Model', 'Accuracy', 'roc_value', 'threshold'])
reports= []
df_Results.sort_values('Methodology', inplace=True)
def display_results():
    global df_Results
    display_text.delete(1.0, tk.END)
    display_text.insert(tk.END, df_Results.to_string())

def evaluate_multiple2():
    # Load the input data file
    global df_Results


    threads = []

    X_train, X_test, y_test, y_train = None, None, None, None
    df = pd.read_csv(input_file_entry.get())
    label = dataLabel_entry.get()
    y = df[label]
    X = df.drop([label], axis=1)
    cols = list(X.columns.values)
    algorithm = algorithm_variable.get()
    crossvalidation = crossvalidation_variable.get()
    balancing = balancing_variable.get()
    Methodology = Methodology_Name(balancing, crossvalidation)
    model_Dict = {
        "Logistic Regression L2": EvaluateLogisticModelsL2,
        "Logistic Regression L1": EvaluateLogisticModelsL1,

        "SVM": EvaluateSVMModels,
        "Isolation Forest": EvaluateIsolationForestModels,
        "KNN": EvaluateKNNModels,
        "XGBoost": EvaluateXGBoostModels
    }
    CV_dict = {
        "RepeatedKFold": RepeatedKFold(n_splits=5, n_repeats=10, random_state=None),
        "StratifiedKFold": StratifiedKFold(n_splits=5, random_state=None),
    }
    Balancing_dict = {
        "Smote": SMOTE(sampling_strategy=0.5),
        "RandomUnderSampler": RandomUnderSampler(sampling_strategy=0.5),
        "RandomOverSampler": RandomOverSampler(sampling_strategy=0.5),
        "None": None
    }
    apool = mpool.ThreadPool(5)
    bpool = mpool.ThreadPool(12)


    lowpowerModels = model_Dict
    lowpowerModels.pop("SVM")
    lowpowerModels.pop("XGBoost")
    hipowerModels={"SVM": EvaluateSVMModels, "XGBoost": EvaluateXGBoostModels}
    for cvname, cv in CV_dict.items():
        for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]
            X_test = X.loc[test_index]
            y_test = y.loc[test_index]
        for modelname, model in lowpowerModels.items():
            Methodology = Methodology_Name("None", cvname)
            print("eh macerena")
            bpool.apply_async(func=model,
                              args=(Methodology, X_train, y_train, X_test, y_test,
                                    False))

    bpool.close()
    bpool.join()
    for cvname, cv in CV_dict.items():
        for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]
            X_test = X.loc[test_index]
            y_test = y.loc[test_index]
        Methodology = Methodology_Name("None", cvname)
        model=hipowerModels.get("XGBoost")
        print("eh macerena2.0")
        model(Methodology, X_train, y_train, X_test, y_test,
              False)
    apool.close()
    apool.join()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(label, axis=1), df[label], test_size=0.3,
                                                        random_state=42)
    model = hipowerModels.get("SVM")
    print("eh macerena")
    sampler=Balancing_dict.get("RandomUnderSampler")
    X_train_over, y_train_over = sampler.fit_resample(X_train, y_train)
    Methodology = Methodology_Name("RandomUnderSampler", "None")
    model(Methodology, X_train_over, y_train_over, X_test, y_test,
                            False)

def print_reports():
    global reports
    display_text.delete(1.0, tk.END)
    for results in reports:
        display_text.insert(tk.END, '\n\nperformance Report for ')
        display_text.insert(tk.END, results['name'])
        display_text.insert(tk.END, "\n\n")
        display_text.insert(tk.END, results['classification_report'])
        display_text.insert(tk.END, 'Accuracy:\n')
        display_text.insert(tk.END, results['accuracy'])
        display_text.insert(tk.END, '\nRoc_value:\n')
        display_text.insert(tk.END, results['roc'])

def evaluate_single():
    # Load the input data file
    global df_Results
    X_train, X_test, y_test, y_train = None, None, None, None
    df = pd.read_csv(input_file_entry.get())
    label = dataLabel_entry.get()
    y = df[label]
    X = df.drop([label], axis=1)
    cols = list(X.columns.values)
    algorithm = algorithm_variable.get()
    crossvalidation = crossvalidation_variable.get()
    balancing = balancing_variable.get()
    Methodology = Methodology_Name(balancing, crossvalidation)
    model_Dict = {
        "Logistic Regression L2": EvaluateLogisticModelsL2,
        "Logistic Regression L1": EvaluateLogisticModelsL1,
        "SVM": EvaluateSVMModels,
        "Isolation Forest": EvaluateIsolationForestModels,
        "KNN": EvaluateKNNModels,
        "XGBoost": EvaluateXGBoostModels
    }
    CV_dict = {
        "RepeatedKFold": RepeatedKFold(n_splits=5, n_repeats=10, random_state=None),
        "StratifiedKFold": StratifiedKFold(n_splits=5, random_state=None),
        "None": None
    }
    Balancing_dict = {
        "Smote": SMOTE(sampling_strategy=0.5),
        "RandomUnderSampler": RandomUnderSampler(sampling_strategy=0.5),
        "RandomOverSampler": RandomOverSampler(sampling_strategy=0.5),
        "None": None
    }

    model = model_Dict.get(algorithm)
    cv = CV_dict.get(crossvalidation)
    sampler = Balancing_dict.get(balancing)

    if crossvalidation == "None" and balancing == "None":
        print("nons")
        X_train, X_test, y_train, y_test = train_test_split(df.drop(label, axis=1), df[label], test_size=0.3,
                                                            random_state=42)
        print(X_test)
        print(y_test)
        results = model(Methodology, X_train, y_train, X_test, y_test, True)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, 'performance Report\n')
        display_text.insert(tk.END, results['classification_report'])
        display_text.insert(tk.END, 'Accuracy:\n')
        display_text.insert(tk.END, results['accuracy'])
        display_text.insert(tk.END, '\nRoc_value:\n')
        display_text.insert(tk.END, results['roc'])
    elif crossvalidation != "None" and balancing != "None":
        print("bofa")
        for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]
            X_test = X.loc[test_index]
            y_test = y.loc[test_index]
            X_train_over, y_train_over = sampler.fit_resample(X_train, y_train)
        X_train_over = pd.DataFrame(data=X_train_over, columns=cols)
        results = model(Methodology, X_train_over, y_train_over, X_test, y_test, True)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, 'performance Report\n')
        display_text.insert(tk.END, results['classification_report'])
        display_text.insert(tk.END, 'Accuracy:\n')
        display_text.insert(tk.END, results['accuracy'])
        display_text.insert(tk.END, '\nRoc_value:\n')
        display_text.insert(tk.END, results['roc'])
    if crossvalidation == "None" and balancing != "None":
        print("onse")
        X_train, X_test, y_train, y_test = train_test_split(df.drop(label, axis=1), df[label], test_size=0.3,
                                                            random_state=42)
        X_train_over, y_train_over = sampler.fit_resample(X_train, y_train)
        results = model(Methodology, X_train_over, y_train_over, X_test, y_test, True)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, 'performance Report\n')
        display_text.insert(tk.END, results['classification_report'])
        display_text.insert(tk.END, 'Accuracy:\n')
        display_text.insert(tk.END, results['accuracy'])
        display_text.insert(tk.END, '\nRoc_value:\n')
        display_text.insert(tk.END, results['roc'])
    if crossvalidation != "None" and balancing == "None":
        print("onse")
        for train_index, test_index in cv.split(X, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
            y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        results = model(Methodology, X_train_cv, y_train_cv, X_test_cv, y_test_cv, True)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, 'performance Report\n')
        display_text.insert(tk.END, results['classification_report'])
        display_text.insert(tk.END, 'Accuracy:\n')
        display_text.insert(tk.END, results['accuracy'])
        display_text.insert(tk.END, '\nRoc_value:\n')
        display_text.insert(tk.END, results['roc'])

display_label = tk.Label(root, text="Output:")
display_label.pack()
display_text = tk.Text(root, height=20, width=300)
display_text.pack()

# Create a button for starting the outlier detection
Evaluate = tk.Button(root, text="Evaluate", command=evaluate_single)
Evaluate.pack()
EvaluateMult = tk.Button(root, text="Run multiple algorithms for evaluation", command=evaluate_multiple2)
EvaluateMult.pack()


def show_results():
    display_text.delete(1.0, tk.END)
    display_text.insert(tk.END, df_Results.to_string())


ShowResults = tk.Button(root, text="Show results table", command=show_results)
ShowResults.pack()
ShowReports = tk.Button(root, text="Show reports for recent evaluations", command=print_reports)
ShowReports.pack()
root.mainloop()