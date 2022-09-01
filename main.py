import pandas as pd

from utils import train_catboost_classifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    target = 'compliance_2021'
    dataset = pd.read_csv('data/inputs/train.csv')

    # # split data into train and test
    # train = dataset.sample(frac=0.6)
    # test = dataset.drop(train.index)

    # split stratified with sklearn
    train, test = train_test_split(dataset, test_size=0.33, stratify=dataset[target])

    # TODO: explore better splitting
    # TODO: data cleaning strategy?  (CATBOOST doesnt need it)
    # TODO: optimize hyperparams, grid search?
    # TODO: add cross validation?
    # TODO: adapt pipeline to produce output file based on "test" file in expected format

    model = train_catboost_classifier(train, test, target=target)
