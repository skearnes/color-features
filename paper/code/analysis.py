"""Analyze results.

For MUV, we should report the datasets separately, I think, but that makes for
a very large table. Either way, we treat each reference molecule as a different
dataset, since we care about the performance changes with respect to that
reference molecule (a separate experiment with different features, just like a
different fingerprint).
"""

import cPickle as pickle
import gflags as flags
import gzip
import logging
import numpy as np
import os
import pandas as pd
from scipy import stats
from statsmodels.stats import proportion
import sys

flags.DEFINE_string('root', None, 'Root directory containing model results.')
flags.DEFINE_string('dataset_file', None, 'Filename containing datasets.')
flags.DEFINE_string('prefix', None, 'Dataset prefix.')
flags.DEFINE_boolean('tversky', False, 'If True, use Tversky features.')
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


def critical_value(num_datasets):
    """Get the critical value for a two-sided 95% confidence interval.

    See http://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
    """
    df = num_datasets - 1  # Degrees of freedom = N - 1.
    return stats.t.isf(0.975, df)


def confidence_interval(delta, metric, sign_test=True):
    """Calculate a two-sided 95% confidence interval for differences."""
    # Wilson score interval for sign test.
    if sign_test:
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
            ci = r'(%.0f, %.0f)' % (lower, upper)
        if lower < 0.5 and upper < 0.5:
            median = r'\bfseries \color{red} ' + median
            ci = r'\bfseries \color{red} ' + ci
        elif lower > 0.5 and upper > 0.5:
            median = r'\bfseries ' + median
            ci = r'\bfseries ' + ci
        return median, ci
    # Paired t-test.
    else:
        # Make sure differences are approximately normally distributed.
        _, p = stats.normaltest(delta)
        if p < 0.05:
            import IPython
            IPython.embed()
            raise AssertionError('Differences are not normally distributed: %g' % p)

        num_datasets = len(delta)
        t_star = np.abs(critical_value(num_datasets))
        mean = delta.mean()
        std = delta.std()
        lower = mean - t_star * std / np.sqrt(num_datasets)
        upper = mean + t_star * std / np.sqrt(num_datasets)
        return mean, str_confidence_interval(lower, upper, metric)


def str_confidence_interval(lower, upper, metric):
    if metric == 'auc':
        number = '%.3f'
    else:
        number = '%.0f'
    ci = r'(\num{%s}, \num{%s})' % (number % lower, number % upper)
    if lower < 0 and upper < 0:
        return r'\bfseries \color{red} ' + ci
    elif lower > 0 and upper > 0:
        return r'\bfseries ' + ci
    else:
        return ci


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
    elif FLAGS.prefix == 'dude':
        subsets = ['xtal', 'omega1']
    elif FLAGS.prefix == 'chembl':
        subsets = ['omega1']
    else:
        raise ValueError(FLAGS.prefix)
    data = []
    models = ['logistic', 'random_forest', 'svm']
    for model in models:
        for subset in subsets:
            df = load_data(model, subset)
            df['model'] = model
            df['subset'] = subset
            data.append(df)
    data = pd.concat(data)
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
