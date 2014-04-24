# coding: utf-8
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def naive_bayes_classifier(ds, training_set):
    """
        Builds the classifier given the Dataset.
        Dataset is splited using cross-validation with 80-20 proportion.

        ds (Dataset) -> GaussianNB
    """
    classifier = GaussianNB()
    classifier.fit(training_set.data, training_set.target)
    return classifier

def classify(classifier, test_set):

    """
        Given my classifier and test_set,
        Evaluate my score and give me a Confusion Matrix.

        classifier (GaussianNB), test (Dataset) -> ConfusionMatrix
    """
    predictions = classifier.predict(test_set.data)

    return confusion_matrix(test_set.target, predictions)
