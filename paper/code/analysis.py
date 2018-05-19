"""Analyze results.

Use the saved model output to calculate AUC and other metrics.
"""

import collections
import cPickle as pickle
import gflags as flags
import gzip
import logging
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from statsmodels.stats import proportion
import sys

flags.DEFINE_string('root', None, 'Root directory containing model results.')
flags.DEFINE_string('dataset_file', None, 'Filename containing datasets.')
flags.DEFINE_string('prefix', None, 'Dataset prefix.')
flags.DEFINE_boolean('tversky', False, 'If True, use Tversky features.')
flags.DEFINE_integer('num_folds', 5, 'Number of cross-validation folds.')
flags.DEFINE_boolean('cycle', False,
                     'If True, expect multiple query molecules.')
flags.DEFINE_string('reload', None, 'Load previously analyzed results.')
flags.DEFINE_string('subset', None, 'Subset.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)


FEATURES_MAP = {
    'rocs': 'TanimotoCombo',
    'shape_color': 'ST-CT',
    'shape_color_components': 'ST-CCT',
    'shape_color_overlaps': 'ST-CAO',
    'shape_color_components_overlaps': 'ST-CCT-CAO',
    'rocs_tversky': 'TverskyCombo',
    'shape_color_tversky': 'STv-CTv',
    'shape_color_components_tversky': 'STv-CCTv',
    'shape_color_components_tversky_overlaps': 'STv-CCTv-CAO',
}

MODEL_MAP = {
    'logistic': 'LR',
    'random_forest': 'RF',
    'svm': 'SVM',
}


def roc_enrichment(fpr, tpr, target_fpr):
    """Get ROC enrichment."""
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0)
    return np.true_divide(np.interp(target_fpr, fpr, tpr), target_fpr)


def get_cv_metrics(y_true, y_pred):
    """Get 5-fold mean AUC."""
    assert len(y_true) == len(y_pred)
    fold_metrics = collections.defaultdict(list)
    for yt, yp in zip(y_true, y_pred):
        assert len(yt) == len(yp)
        fold_metrics['auc'].append(metrics.roc_auc_score(yt, yp))
        fpr, tpr, _ = metrics.roc_curve(yt, yp)
        for x in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            fold_metrics['e-%g' % x].append(roc_enrichment(fpr, tpr, x))
    return fold_metrics


def add_rows(features, scores, rows, dataset, index=None):
    """Record per-fold and averaged cross-validation results."""
    for fold in range(len(scores['auc'])):
        row = {'dataset': dataset, 'features': features, 'fold': fold}
        if index is not None:
            row['index'] = index
        for key, values in scores.iteritems():
            row[key] = values[fold]
        rows.append(row)

    # Averages
    row = {'dataset': dataset, 'features': features, 'fold': 'all'}
    if index is not None:
        row['index'] = index
    for key, values in scores.iteritems():
        row[key] = np.mean(values)
    rows.append(row)


def load_output_and_calculate_metrics(model, subset):
    """Calculate metrics using saved model output.

    Args:
        model: String model type (e.g. logistic).
        subset: String query subset (e.g. omega1).

    Returns:
        DataFrame containing calculated metrics for each model/subset, including
        per-fold and average values for each reference molecule.
    """
    with open(FLAGS.dataset_file) as f:
        datasets = [line.strip() for line in f]
    rows = []
    for dataset in datasets:
        ref_idx = 0
        while True:  # Cycle through reference molecules.
            ref_idx_exists = get_ref_rows(model, subset, dataset, ref_idx, rows)
            if not FLAGS.cycle or not ref_idx_exists:
                break
            ref_idx += 1
        logging.info('%s\t%d', dataset, ref_idx)
    return pd.DataFrame(rows)


def get_ref_rows(model, subset, dataset, ref_idx, rows):
    logging.debug('ref_idx %d', ref_idx)
    for features in FEATURES_MAP.keys():
        logging.debug('Features: %s', features)
        fold_y_true = []
        fold_y_pred = []
        for fold_idx in range(FLAGS.num_folds):
            filename = get_output_filename(dataset, model, subset, features,
                                           fold_idx, ref_idx)
            if not os.path.exists(filename):
                return False
            logging.debug(filename)
            with gzip.open(filename) as f:
                df = pickle.load(f)
            fold_y_true.append(df['y_true'].values)
            fold_y_pred.append(df['y_pred'].values)
        scores = get_cv_metrics(fold_y_true, fold_y_pred)
        add_rows(features, scores, rows, dataset, index=ref_idx)
    return True


def get_output_filename(dataset, model, subset, features, fold_idx, ref_idx):
    if FLAGS.cycle:
        filename = os.path.join(
            '%s-%s' % (FLAGS.root, subset),
            dataset,
            'fold-%d' % fold_idx,
            '%s-%s-%s-%s-%s-fold-%d-ref-%d-output.pkl.gz' % (
                FLAGS.prefix, dataset, model, subset, features,
                fold_idx, ref_idx))
    else:
        assert ref_idx == 0
        filename = os.path.join(
            '%s-%s' % (FLAGS.root, subset),
            dataset,
            'fold-%d' % fold_idx,
            '%s-%s-%s-%s-%s-fold-%d-output.pkl.gz' % (
                FLAGS.prefix, dataset, model, subset, features,
                fold_idx))
    return filename


def load_data(model, subset):
    data = []
    with open(FLAGS.dataset_file) as f:
        for line in f:
            dataset = line.strip()
            filename = os.path.join(FLAGS.root, '%s-%s-%s-%s.pkl.gz' % (
                                    FLAGS.prefix, dataset, model, subset))
            assert os.path.exists(filename)
            logging.info(filename)
            with gzip.open(filename) as g:
                df = pickle.load(g)
            data.append(df)
    return pd.concat(data)


def confidence_interval(delta, metric):
    """Calculate a two-sided 95% confidence interval for differences."""
    # Wilson score interval for sign test.
    num_successes = np.count_nonzero(delta > 0)
    num_trials = np.count_nonzero(delta != 0)  # Exclude zero differences.
    lower, upper = proportion.proportion_confint(
        num_successes, num_trials, alpha=0.05, method='wilson')
    median_delta = delta.median()
    if metric == 'auc':
        median = r'%.3f' % median_delta
        ci = r'(%.2f, %.2f)' % (lower, upper)
    else:
        median = r'%.0f' % median_delta
        ci = r'(%.2f, %.2f)' % (lower, upper)
    if lower < 0.5 and upper < 0.5:
        median = r'\bfseries \color{red} ' + median
        ci = r'\bfseries \color{red} ' + ci
    elif lower > 0.5 and upper > 0.5:
        median = r'\bfseries ' + median
        ci = r'\bfseries ' + ci
    return median, ci


def data_table(data, subsets, models, kind=None, tversky=False):
    """Get medians and compare everything to ROCS.

    Args:
        data: DataFrame containing model performance.
        subsets: List of query subsets.
        models: List of models to include in the table.
        kind: List of metrics to report. Defaults to ['auc'].
        tversky: Boolean whether to use Tversky features. If False, use Tanimoto
            features.
    """
    if kind is None:
        kind = ['auc']
    if tversky:
        rocs_baseline = 'rocs_tversky'
        features_order = ['shape_color_tversky',
                          'shape_color_components_tversky',
                          'shape_color_overlaps',
                          'shape_color_components_tversky_overlaps']
    else:
        rocs_baseline = 'rocs'
        features_order = ['shape_color', 'shape_color_components',
                          'shape_color_overlaps',
                          'shape_color_components_overlaps']
    table = []

    # Get ROCS row.
    row = [r'\cellcolor{white} ROCS', FEATURES_MAP[rocs_baseline]]

    for subset in subsets:
        rocs_mask = ((data['features'] == rocs_baseline) &
                     (data['subset'] == subset) &
                     (data['model'] == models[0]))
        rocs_df = data[rocs_mask]
        logging.info('Confidence interval N = %d', len(rocs_df))
        logging.info('Number of datasets = %d',
                     len(pd.unique(rocs_df['dataset'])))
        for metric in kind:
            if metric == 'auc':
                number = '%.3f'
            else:
                number = '%.0f'
            row.extend([number % rocs_df[metric].median(), '', ''])
    table.append(' & '.join(row))

    # Get model rows.
    for model in models:
        for features in features_order:
            if features == features_order[-1]:
                row = [r'\multirow{-%d}{*}{\cellcolor{white} %s}' % (
                    len(features_order), MODEL_MAP[model])]
            else:
                row = [r'\cellcolor{white}']
            row.append(FEATURES_MAP[features])
            for subset in subsets:
                mask = ((data['features'] == features) &
                        (data['subset'] == subset) &
                        (data['model'] == model))
                df = data[mask]
                rocs_mask = ((data['features'] == rocs_baseline) &
                             (data['subset'] == subset) &
                             (data['model'] == model))
                rocs_df = data[rocs_mask]
                for metric in kind:
                    if metric == 'auc':
                        number = '%.3f'
                    else:
                        number = '%.0f'
                    row.append(number % df[metric].median())
                    if features == rocs_baseline:
                        row.append('')
                        row.append('')
                    else:
                        assert np.array_equal(df['dataset'].values,
                                              rocs_df['dataset'].values)
                        if 'index' in df.columns:
                            assert np.array_equal(df['index'].values,
                                                  rocs_df['index'].values)
                        delta = df.copy()
                        delta[metric] -= rocs_df[metric].values
                        row.extend(confidence_interval(delta[metric], metric))
            table.append(' & '.join(row))
    print ' \\\\\n'.join(table)


def main():
    if FLAGS.prefix == 'muv':
        subsets = ['omega1']
        assert FLAGS.cycle
    elif FLAGS.prefix == 'dude':
        subsets = ['xtal', 'omega1']
    elif FLAGS.prefix == 'chembl':
        subsets = ['omega1']
        assert FLAGS.cycle
    else:
        raise ValueError(FLAGS.prefix)
    if FLAGS.subset is not None:
        subsets = [FLAGS.subset]

    # Load data from output or previously processed.
    models = ['logistic', 'random_forest', 'svm']
    if FLAGS.reload is not None:
        logging.info('Loading processed data from %s', FLAGS.reload)
        data = pd.read_pickle(FLAGS.reload)
    else:
        data = []
        for model in models:
            for subset in subsets:
                logging.info('%s\t%s', model, subset)
                df = load_output_and_calculate_metrics(model, subset)
                df['model'] = model
                df['subset'] = subset
                data.append(df)
        data = pd.concat(data)

        # Save processed data.
        filename = '%s-processed.pkl.gz' % FLAGS.prefix
        logging.info('Saving processed data to %s', filename)
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # Only keep 5-fold mean information.
    mask = data['fold'] == 'all'
    data = data[mask]

    # AUC tables.
    # Combine subsets into a single table here.
    logging.info('AUC table')
    data_table(data, subsets, models, kind=['auc'], tversky=FLAGS.tversky)

    # Enrichment tables.
    # One per FPR.
    for metric in ['e-0.005', 'e-0.01', 'e-0.02', 'e-0.05']:
        logging.info('Metric: %s', metric)
        logging.info('Enrichment table')
        data_table(data, subsets, models, kind=[metric], tversky=FLAGS.tversky)

if __name__ == '__main__':
    flags.MarkFlagAsRequired('root')
    flags.MarkFlagAsRequired('dataset_file')
    flags.MarkFlagAsRequired('prefix')
    FLAGS(sys.argv)
    main()
