import re
import pandas as pd
import openai
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from difflib import get_close_matches


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to get OpenAI API response for a given query
def get_openai_response(query):
    openai.api_key = 'your_openai_api_key'
    response = openai.Completion.create(
        engine="davinci",
        prompt=query,
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

def healthcare_chatbot(training_file, testing_file, symptom_description_file, symptom_severity_file, symptom_precaution_file):
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
            print("You should take the consultation from a doctor.")
        else:
            print("It might not be that bad but you should take precautions.")

    def getInfo():
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("\nYour Name? \t\t\t\t", end="->")
        name = input("")
        print("Hello, ", name)

    def check_pattern(dis_list, inp):
        pattern = re.compile(r'\b' + re.escape(inp) + r'\b', re.IGNORECASE)
        pred_list = [symptom for symptom in dis_list if re.search(pattern, symptom)]
        if pred_list:
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

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        while True:

            print("\nEnter the symptom you are experiencing  \t\t", end="->")
            disease_input = input("")
            conf, cnf_dis = check_pattern(chk_dis, disease_input)
            if conf == 1:
                print("searches related to input: ")
                for num, it in enumerate(cnf_dis):
                    print(num, ")", it)
                if num != 0:
                    print(f"Select the one you meant (0 - {num}):  ", end="")
                    conf_inp = int(input(""))
                else:
                    conf_inp = 0

                disease_input = cnf_dis[conf_inp]
                break
            else:
                print("Enter a valid symptom.")

        while True:
            try:
                num_days = int(input("Okay. From how many days ? : "))
                break
            except:
                print("Enter a valid input.")
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])

                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

                print("Are you experiencing any ")
                symptoms_exp = []
                for syms in list(symptoms_given):
                    inp = ""
                    print(syms, "? : ", end='')
                    while True:
                        inp = input("")
                        if(inp == "yes" or inp == "no"):
                            break
                        else:
                            print("Provide proper answers i.e. (yes/no) : ", end="")
                    if(inp == "yes"):
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp)
                calc_condition(symptoms_exp, num_days)
                if(present_disease[0] == second_prediction[0]):
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])

                else:
                    print("You may have ", present_disease[0], "or ", second_prediction[0])
                    print(description_list[present_disease[0]])
                    print(description_list[second_prediction[0]])

                precution_list = precautionDictionary[present_disease[0]]
                print("Take the following measures : ")
                for i, j in enumerate(precution_list):
                    print(i + 1, ")", j)

                # If the disease is not in the dataset, use OpenAI to generate a response
                if present_disease[0] == "undefined!":
                    query = f"I have never heard of '{disease_input}'. Can you provide more details?"
                    openai_response = get_openai_response(query)
                    print("OpenAI Response: ", openai_response)

        recurse(0, 1)

    getInfo()
    tree_to_code(clf, cols)
    print("----------------------------------------------------------------------------------------")

# Example usage:
healthcare_chatbot('Data/Training.csv', 'Data/Testing.csv', 'MasterData/symptom_Description.csv', 'MasterData/symptom_severity.csv', 'MasterData/symptom_precaution.csv')