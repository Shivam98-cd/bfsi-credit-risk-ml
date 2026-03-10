import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, \
    confusion_matrix


def calculate_ks_stat(y_true, y_prob):
    """Calculates the KS Statistic Table"""
    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    df['decile'] = pd.qcut(df['prob'], 10, labels=False, duplicates='drop')

    ks = df.groupby('decile').apply(lambda x: pd.Series({
        'bad': x['target'].sum(),
        'good': len(x) - x['target'].sum()
    })).reset_index().sort_values('decile', ascending=False)

    ks['cum_bad_pct'] = (ks['bad'].cumsum() / ks['bad'].sum()) * 100
    ks['cum_good_pct'] = (ks['good'].cumsum() / ks['good'].sum()) * 100
    ks['KS'] = abs(ks['cum_bad_pct'] - ks['cum_good_pct'])
    return ks


def get_full_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "KS_Table": calculate_ks_stat(y_test, y_prob)
    }

def get_feature_importance(model, feature_names):
    """Extracts feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        return feature_importance_df
    return None
