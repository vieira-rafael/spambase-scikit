# coding> utf-8

from datatypes import Dataset

def load_spam_ds():
    """
    Loads the data from file and build the dataset in scikit format.

    () -> NamedTuple(Dataset)
    """

    data = []
    target = []

    with open("data/spambase.data", "r") as f:
        for line in f:
            # Removes \r\n from line
            line = line[:-2]
            
            items = line.split(",")
            features = items[:-1]
            spam_class = items[-1]
            data.append(features)
            target.append(spam_class)

    return Dataset(data, target)
    
if __name__ == "__main__":
    ds = load_spam_ds()
