# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:58:52 2017
@author: NishitP

Modified by Sambhav Jain:
- Stripped down to only train and save the Logistic Regression (n-gram/TF-IDF) 
  pipeline, since that is the model actually used by prediction.py.
- Removed the Random Forest + GridSearchCV hyperparameter search sections, 
  which were causing an OpenBLAS memory allocation crash on this machine and 
  were not required to produce final_model.sav.
- Removed unused bag-of-words-only classifiers, learning curve plots, and 
  precision-recall plots (exploratory code not needed for the saved model).
- Kept the K-Fold cross-validation check so real performance numbers are 
  still printed.
"""

import DataPrep
import FeatureSelection
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ------------------------------------------------------------------
# Train the Logistic Regression pipeline using TF-IDF + n-grams
# (This is the model prediction.py loads and uses)
# ------------------------------------------------------------------
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf', FeatureSelection.tfidf_ngram),
        ('LogR_clf', LogisticRegression(penalty="l2", C=1, max_iter=1000))
        ])

logR_pipeline_ngram.fit(DataPrep.train_news['Statement'], DataPrep.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(DataPrep.test_news['Statement'])
test_accuracy = np.mean(predicted_LogR_ngram == DataPrep.test_news['Label'])
print(f"\nTest set accuracy: {test_accuracy:.4f}")

print("\nClassification report (test set):")
print(classification_report(DataPrep.test_news['Label'], predicted_LogR_ngram))

# ------------------------------------------------------------------
# K-Fold cross-validation, so we have a genuine cross-validated score
# ------------------------------------------------------------------
def build_confusion_matrix(classifier):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_ind, test_ind in k_fold.split(DataPrep.train_news):
        train_text = DataPrep.train_news.iloc[train_ind]['Statement']
        train_y = DataPrep.train_news.iloc[train_ind]['Label']

        test_text = DataPrep.train_news.iloc[test_ind]['Statement']
        test_y = DataPrep.train_news.iloc[test_ind]['Label']

        classifier.fit(train_text, train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='True' if 'True' in test_y.unique() else test_y.unique()[0])
        scores.append(score)

    print('\nTotal statements classified:', len(DataPrep.train_news))
    print('Average F1 score:', sum(scores) / len(scores))
    print('Confusion matrix:')
    print(confusion)


print("\nRunning 5-fold cross-validation on Logistic Regression (n-gram/TF-IDF)...")
build_confusion_matrix(logR_pipeline_ngram)

# ------------------------------------------------------------------
# Save the trained model to disk so prediction.py can load it
# ------------------------------------------------------------------
model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram, open(model_file, 'wb'))
print(f"\nSaved trained model to {model_file}")