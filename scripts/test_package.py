import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV

from ImbalancedEstimator.imbalanced_logistic_regression import ImbalancedLogisticRegression


# import sys
# import os
# sys.path.extend([os.path.join(os.path.dirname(__file__), '..')])


def test_package(n_samples=10000, class_sep=0.9, percent_minority_class=0.05, n_features=2, random_state=22):
    X, y = make_classification(n_samples=n_samples,
                               class_sep=class_sep,
                               # The lower this value is, the closer together points from different classes will be.
                               flip_y=0,
                               # defines the probability that the target variable for a sample will be flipped (it becomes 1 when it should be 0 and vice-versa).
                               n_clusters_per_class=1,
                               n_classes=2,
                               n_features=n_features,  # Dimension of the data
                               n_informative=n_features,  # Number of informative features (not redundant or repeated)
                               n_redundant=0,  # 0 of the features will just be combinations of other features
                               n_repeated=0,  # 0 of the features will be duplicates
                               weights=[1 - percent_minority_class, percent_minority_class],
                               # 99% of the targets will be 0, 1% will be 1.
                               random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Create dataset and run the model
    imbalanced_lr = ImbalancedLogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                 intercept_scaling=1, max_iter=100,
                                                 multi_class='auto', penalty='l2',
                                                 random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                                                 warm_start=False)

    param_grid = [{'C': [10 ** int(C) for C in np.arange(-5, 5)],
                   'penalty': ['l2', 'l1']}]
    gs = GridSearchCV(estimator=imbalanced_lr, param_grid=param_grid, scoring=ImbalancedLogisticRegression.score, cv=2,
                      n_jobs=1)

    gs.fit(X_train, y_train)

    print('The best model is: ${}$'.format(gs.best_estimator_))
    print('The best hyperparameters are: ${}$'.format(gs.best_params_))
    print('The score of the best model on the validation set is: ${:.3f}$'.format(gs.best_score_))
    print('Run the best model on the test: ${:.3f}$'.format(
        gs.score(X_test, y_test)))  # Run with the best combination of hyperparameters


class myArgparserFormater(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    """
    RawTextHelpFormatter: can break lines in the help text, but don't print default values
    ArgumentDefaultsHelpFormatter: print default values, but don't break lines in the help text
    """
    pass


def parse_args():
    help_txt = "Acurate assembly of transcripts according mapped reads"
    parser = argparse.ArgumentParser(description=help_txt, formatter_class=myArgparserFormater)

    parser.add_argument('--n-samples', help='Size of the test sample', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    # sys.path.extend(['C:\\Users\\Refael Kohen\\package_module\\packaging\\'])

    print(sys.path)
    args = parse_args()

    print('Your parameter is {}'.format(args.n_samples))

    test_package(int(args.n_samples))