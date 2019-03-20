import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import tree
from sklearn.metrics import roc_curve, auc

# pd.set_option('display.max_columns', 1220) per visualizzare tutte le colonne del dataset

# Importo il dataset
data = open('survey_results_public.csv')
dataset = pd.read_csv(data, index_col=0)
print("Dimensioni iniziali dataset: ", dataset.shape)

# Dataset setup:
#  1)eliminare righe con NaN nella colonna Salario

dataset = dataset.dropna(subset=['Salary'])
print("Dimensioni dataset dopo rimozione NaN nella colonna Salario: ", dataset.shape)

#  2)estrarre la colonna Salario

salary = dataset.pop('Salary')

#  3)eliminare colonne irrilevanti

dataset = dataset.drop(['ExpectedSalary', 'NonDeveloperType'], axis=1)
print("Dimensione dataset dopo rimozione colonne irrilevanti e la colonna Salario: ", dataset.shape)

#  4)trasformare le colonne con stringhe in tante colonne binarie, separando anche quelle con valori sovrapposti.
#    I valori NaN vengono trasformati in stringa per essere utilizzati nella previsione.

columns = dataset.columns.values
is_num = np.array([col for col in dataset.dtypes != 'object'])
col_num = columns[is_num]
col_obj = columns[~is_num]

for col in col_obj:
    dataset[col].fillna(value="NaN", inplace=True)

for col in col_obj:
    tdf = dataset[col].str.get_dummies(sep='; ')
    tdf.columns = [col+'_'+s for s in tdf.columns]
    dataset = dataset.join(tdf).drop(col, axis=1)

print("Dimensione dataset dopo lo split delle colonne di stringhe: ", dataset.shape)

#  5)sostituire i valori NaN nelle colonne numeriche, inserendo al loro posto la mediana.

for col in col_num:
    dataset[col].fillna(value=dataset[col].dropna().median(), inplace=True)

# Trovo la mediana dei valori nella colonna Salario

mediana = salary.median()
print("\nMediana dei salari: ", mediana)

# Divido il dataset in train-set e test-set

RANDOM_STATE = 42
TEST_SIZE = 0.2

target = salary >= mediana
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Traccio il grafico a seconda delle veriazioni del max_depth e ne trovo il massimo secondo la AUC metric

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    dt = tree.DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

max_depth = 0
max_value = test_results[0]
for i in range(1, len(test_results)):
    if test_results[i] > max_value:
        max_value = test_results[i]
        max_depth = i+1

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()
print("max depth: ", max_depth)

# Definisco il Decision Tree e stampo la precisione

clf = tree.DecisionTreeClassifier(max_depth=max_depth)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Predizione del test set:\n", y_pred, "\nScore: ", clf.score(x_test, y_test))

# AUC metric

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC score: ", roc_auc)

# 10-fold cross validation con Decision Tree

kf = KFold(n_splits=10)
k_fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(clf, x_train, y_train, cv=k_fold)
print('10-fold cross validation score:\n {}'.format(scores))
print('Media: {}'.format(scores.mean()))

# Creo il grafo dell'albero

'''
import pydotplus
import collections
import graphviz

dot_data = tree.export_graphviz(clf, feature_names=dataset.columns.values, out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('DecisionTree.png')
'''
