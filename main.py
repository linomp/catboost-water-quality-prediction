from datetime import datetime

import pandas as pd
import pickle

from utils import train_catboost_classifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    target = 'compliance_2021'

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

    except FileNotFoundError:
        dataset = pd.read_csv('data/inputs/train.csv')

        # stratified split with sklearn
        train, test = train_test_split(dataset, test_size=0.33, stratify=dataset[target])

        model = train_catboost_classifier(train, test, target=target)

        # save the model as pickle
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # predict on test set for competition
    test = pd.read_csv('data/inputs/test.csv')
    test[target] = model.predict(test)

    # produce output file based on "test" file in expected format
    output = test[['station_id', target]]
    output.to_csv(f'data/outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
