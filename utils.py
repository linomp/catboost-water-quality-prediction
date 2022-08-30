import wittgenstein as lw
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from wittgenstein.interpret import interpret_model


def get_catboost_roc_curve(model, X, y, categorical_features_indexes):
    # auc = model.best_score_['validation']['AUC']
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    (fpr, tpr, _) = get_roc_curve(model,
                                  Pool(X, y, cat_features=categorical_features_indexes, feature_names=list(X.columns)))

    ax = plt.gca()
    ax.plot(fpr, tpr, color="b",
            lw=2,
            alpha=0.8,
            label="ROC (AUC = %0.2f)" % auc)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random classifier", alpha=0.8)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")

    plt.show()

    return auc


def get_ruleset_from_catboost(max_rules, max_rule_conds, max_total_conds, k, dl_allowance, n_discretize_bins,
                              prune_size, train, test, full_dataset, columns_to_ignore, activity, output_dir):
    """
    Prototype of interpretable model extraction: CatBoost classifier then Ruleset
    - https://github.com/imoscovitz/wittgenstein#interpreter-models
    - https://catboost.ai/en/docs/concepts/python-usages-examples
    """

    ruleset_params = {"max_rules": max_rules}
    interpreter = lw.RIPPER(**ruleset_params)

    # Fit a 'complex' model - CatBoostClassifier
    X_train = train.drop(['knocked_out_case'], axis=1)
    y_train = train['knocked_out_case']
    X_test = test.drop(['knocked_out_case'], axis=1)
    y_test = test['knocked_out_case']
    categorical_features_indexes = [X_train.columns.get_loc(col) for col in X_train.columns if
                                    X_train[col].dtype == "object"]
    test_pool = Pool(X_test, y_test, cat_features=categorical_features_indexes, feature_names=list(X_test.columns))

    model = CatBoostClassifier(iterations=10, depth=16, loss_function='Logloss', eval_metric="AUC")
    model.fit(X_train, y_train, cat_features=categorical_features_indexes,
              use_best_model=True, eval_set=test_pool)

    # Get Catboost model metrics
    catboost_auc_score = get_catboost_roc_curve(model, X_test, y_test, categorical_features_indexes)

    # interpret with wittgenstein and get a ruleset
    def predict_fn(data, _):
        res = model.predict(data, prediction_type='Class')
        res = [x == 'True' for x in res]
        return res

    interpret_model(model=model, X=X_train, interpreter=interpreter, model_predict_function=predict_fn)
    ruleset_model = interpreter

    catboost_feature_importances = list(
        model.get_feature_importance(prettified=True).itertuples(index=False, name=None))

    ruleset_params['catboost'] = model.get_params()
    ruleset_params['catboost_total_trees'] = model.tree_count_

    return ruleset_model, ruleset_params, model, catboost_feature_importances, catboost_auc_score
