# AIproject
Il codice utilizzato permette di fare uso del _DecisionTree_ di scikit-learn per determinare se un nuovo impiegato avrà uno stipendio maggiore rispetto alla mediana degli stipendi degli impiegati nel dataset.\
Il dataset è reperibile su [Stack Overflow Developer Survey, 2017](https://www.kaggle.com/stackoverflow/so-survey-2017) e deve essere scaricato per il corretto funzionamento del codice.

Sono state usate le librerie:
* __numpy__, __pandas__ per operare sul _dataset_
* __matplotlib__ per disegnare i vari _grafici_
* __sklearn__ per definire il _decision tree_, per creare la _roc curve_ e per utilizzare il _10-fold cross validation_
* __pydotplus__, __graphivz__, __collections__ per disegnare il _grafo_ dell'albero

Nella prima parte del codice viene riadattato il dataset per poter essere letto da DecisionTree, e quindi:
1. eliminazione righe con NaN nella colonna Salary
2. estrazione della colonna Salary
3. eliminazione colonne irrilevanti
4. sostituzione valori NaN
5. separazione delle colonne con risposte sovrapposte 

In seguito è stato definito il target ed è stato diviso il dataset in train-set e test-set per determinare la precisione della previsione dell'albero.\
E' stato inoltre verificato quale fosse il parametro max_depth adatto a seconda del train-set, seguendo le indicazioni dell'articolo  ["InDepth: Parameter tuning for Decision Tree"](https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3) di _Mohtadi Ben Fraj_, illustrandone il grafico rispetto alla sua variazione.\
Infine, dopo aver creato l'albero decisionale ed aver determinato la precisione della previsione mediante il 10-fold cross validation (_circa 0.83_), è stato disegnato il grafo dell'albero, seguendo il codice ottenuto dall'articolo ["Creating and Visualizing Decision Trees with Python"](https://pythonprogramminglanguage.com/decision-tree-visual-example/) di _Russel_.

_Per maggiori chiarimenti si prega di leggere la relazione allegata nella repository._
