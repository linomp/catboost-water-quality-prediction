import shap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve


def get_catboost_roc_auc(model, X, y, categorical_features_indexes):
    accuracy = model.score(X, y)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    (fpr, tpr, _) = get_roc_curve(model,
                                  Pool(X, y, cat_features=categorical_features_indexes, feature_names=list(X.columns)))

    ax = plt.gca()
    ax.plot(fpr, tpr, color="b",
            lw=2,
            alpha=0.8,
            label=f"ROC (AUC = {auc:.2f} | acc = {accuracy:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random classifier", alpha=0.8)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")

    plt.show()

    return auc


def train_catboost_classifier(train, test, target, max_rules=3):
    # Fit a 'complex' model - CatBoostClassifier
    X_train = train.drop([target], axis=1)
    y_train = train[target]
    X_test = test.drop([target], axis=1)
    y_test = test[target]

    categorical_features_indexes = [X_train.columns.get_loc(col) for col in X_train.columns if
                                    X_train[col].dtype == "object"]
    test_pool = Pool(X_test, y_test, cat_features=categorical_features_indexes, feature_names=list(X_test.columns))

    model = CatBoostClassifier(nan_mode='Min',
                               thread_count=8,
                               task_type='CPU',
                               learning_rate=0.01,
                               eval_metric="AUC")

    model.fit(X_train,
              y_train,
              cat_features=categorical_features_indexes,
              use_best_model=True,
              eval_set=test_pool)

    # Get Catboost model metrics

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

    catboost_auc_score = get_catboost_roc_auc(model, X_test, y_test, categorical_features_indexes)

    return model
