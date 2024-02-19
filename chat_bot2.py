import re
import pickle
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def healthcare_chatbot(user_input):
    training_file = 'Data/Training.csv'
    testing_file = 'Data/Testing.csv'
    symptom_description_file = 'MasterData/symptom_Description.csv'
    symptom_severity_file = 'MasterData/symptom_severity.csv'
    symptom_precaution_file = 'MasterData/symptom_precaution.csv'

    training = pd.read_csv(training_file)
    testing = pd.read_csv(testing_file)
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['prognosis']
    y1 = y

    reduced_data = training.groupby(training['prognosis']).max()

    # Mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    scores = cross_val_score(clf, x_test, y_test, cv=3)
    print(scores.mean())

    model = SVC()
    model.fit(x_train, y_train)
    print("for svm: ")
    print(model.score(x_test, y_test))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    severityDictionary = {}
    description_list = {}
    precautionDictionary = {}

    with open(symptom_severity_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                severityDictionary[row[0]] = int(row[1])
        except:
            pass

    with open(symptom_description_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

    with open(symptom_precaution_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    def readn(nstr):
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()

    symptoms_dict = {}

    for index, symptom in enumerate(x):
        symptoms_dict[symptom] = index

    def calc_condition(exp, days):
        sum = 0
        for item in exp:
            sum = sum + severityDictionary[item]
        if ((sum * days) / (len(exp) + 1) > 13):
            return "You should take the consultation from a doctor."
        else:
            return "It might not be that bad but you should take precautions."

    def getInfo():
        return "-----------------------------------HealthCare ChatBot-----------------------------------\n\nYour Name? \t\t\t\t"

    def check_pattern(dis_list, inp):
        pred_list = []
        inp = inp.replace(' ', '_')
        patt = f"{inp}"
        regexp = re.compile(patt)
        pred_list = [item for item in dis_list if regexp.search(item)]
        if(len(pred_list) > 0):
            return 1, pred_list
        else:
            return 0, []

    def sec_predict(symptoms_exp):
        df = pd.read_csv(training_file)
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    def print_disease(node):
        node = node[0]
        val = node.nonzero()
        disease = le.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def tree_to_code(tree, feature_names, user_input):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        disease_input = user_input['symptom']
        num_days = user_input['days']

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if val <= threshold:
                    return recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    return recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])

                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

                symptoms_exp = []
                for syms in list(symptoms_given):
                    if syms == "None":
                        continue
                    else:
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp)
                return calc_condition(symptoms_exp, num_days)

        return recurse(0, 1)

    # Invoke the function and return the response
    return tree_to_code(clf, cols, user_input)

# Example usage:
# user_input = {'symptom': 'fever', 'days': 3}
# print(healthcare_chatbot(user_input))
