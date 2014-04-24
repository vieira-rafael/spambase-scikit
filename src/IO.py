# coding> utf-8

from datatypes import Dataset
from classifier import naive_bayes_classifier, classify
from feature_selection import univariate_feature_selection

from sklearn.cross_validation import train_test_split


def load_spam_ds():
    """
    Loads the data from file and build the dataset in scikit format.

    () -> Dataset
    """

    data = []
    target = []
    i = 0
    with open("data/spambase.data", "r") as f:
        for line in f:
            # Removes \r\n from line
            line = line[:-2]
            
            items = line.split(",")
            features = [float(item) for item in items[:-1]]
            spam_class = int(items[-1])
            data.append(features)
            target.append(spam_class)
    
    return Dataset(data, target)

def split_train_test(ds):
    """
    Given the dataset, split in two datasets:
    One is the Training set. Other is the Test set.
    The proportion is 80% to 20% Respectively
    
    Dataset -> Dataset, Dataset
    """

    samples_train, samples_test, classes_train, classes_test = train_test_split(ds.data, ds.target, test_size=0.2)
    training_set = Dataset(samples_train, classes_train)
    test_set = Dataset(samples_test, classes_test)
    return training_set, test_set

def run(n=0, method=None):
    """
    Starts the classification Pipeline
    """
    ds = load_spam_ds()
    if method:
        ds = univariate_feature_selection(n, ds)
    training_set, test_set = split_train_test(ds)
    classifier = naive_bayes_classifier(ds, training_set)
    cm = classify(classifier, test_set)
    print(cm)

if __name__ == "__main__":
    run()
