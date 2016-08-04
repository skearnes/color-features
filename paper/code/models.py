"""Build models to compare featurizations.

Non-parametric:
1. ROCS TanimotoCombo
2. ROCS RefTversky

Parametric (use both Tanimoto and RefTversky versions):
3. ROCS shape + ROCS color
4. ROCS shape + color components
5. ROCS shape + color components + color atom overlaps

It would be interesting to see if the color atom overlaps track well with
RefTversky.

TODO: Multiple reference molecules for MUV.
"""

import collections
import cPickle as pickle
import gflags as flags
import gzip
from oe_utils.utils import h5_utils
import logging
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import grid_search
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
import sys

flags.DEFINE_string('rocs_actives', None,
                    'ROCS overlays for actives.')
flags.DEFINE_string('rocs_inactives', None,
                    'ROCS overlays for inactives.')
flags.DEFINE_string('color_components_actives', None,
                    'Color components for actives.')
flags.DEFINE_string('color_components_inactives', None,
                    'Color components for inactives.')
flags.DEFINE_string('color_atom_overlaps_actives', None,
                    'Color atom overlaps for actives.')
flags.DEFINE_string('color_atom_overlaps_inactives', None,
                    'Color atom overlaps for inactives.')
flags.DEFINE_string('dataset', None, 'Dataset.')
flags.DEFINE_string('dataset_file', None, 'Filename containing datasets.')
flags.DEFINE_string('model', 'logistic', 'Model type.')
flags.DEFINE_string('prefix', None, 'Prefix for output filenames.')
flags.DEFINE_boolean('skip_failures', True, 'Skip failed datasets.')
flags.DEFINE_boolean('cycle', False, 'If True, use cyclic validation.')
flags.DEFINE_integer('n_jobs', 1, 'Number of parallel jobs.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)

# These datasets failed OMEGA expansion of their xtal ligand.
FAILED = [
    'aa2ar',
    'andr',
    'aofb',
    'bace1',
    'braf',
    'dyr',
    'esr2',
    'fkb1a',
    'kif11',
    'rxra',
    'sahh',
    'urok',
]


def load_datasets():
    datasets = []
    if FLAGS.dataset is not None:
        datasets.append(FLAGS.dataset)
    elif FLAGS.dataset_file is not None:
        with open(FLAGS.datasets) as f:
            for line in f:
                datasets.append(line.strip())
    else:
        raise ValueError('No dataset(s) specified.')
    return datasets


def load_features_and_labels(dataset):
    """Load features from ROCS overlays.

    Returns:
        features: Dict mapping feature names to numpy arrays.
        labels: Numpy array containing labels for molecules.
    """
    features = {}

    # ROCS.
    rocs_actives = h5_utils.load(FLAGS.rocs_actives % dataset)
    rocs_inactives = h5_utils.load(FLAGS.rocs_inactives % dataset)

    num_actives = len(rocs_actives['shape_tanimoto'])
    num_inactives = len(rocs_inactives['shape_tanimoto'])
    labels = np.concatenate((np.ones(num_actives, dtype=int),
                             np.zeros(num_inactives, dtype=int)))

    for feature in ['shape_tanimoto', 'color_tanimoto', 'shape_overlap',
                    'color_overlap', 'ref_self_shape', 'ref_self_color',
                    'fit_self_shape', 'fit_self_color']:
        features[feature] = np.concatenate((rocs_actives[feature],
                                            rocs_inactives[feature]))
    features['combo_tanimoto'] = np.true_divide(
        features['shape_tanimoto'] + features['color_tanimoto'], 2)

    # Tversky.
    features['shape_tversky'] = np.true_divide(
        features['shape_overlap'],
        0.95 * features['ref_self_shape'] +
        0.05 * features['fit_self_shape'])
    features['color_tversky'] = np.true_divide(
        features['color_overlap'],
        0.95 * features['ref_self_color'] +
        0.05 * features['fit_self_color'])
    features['combo_tversky'] = np.true_divide(
        features['shape_tversky'] + features['color_tversky'], 2)

    # Color components.
    cc_actives = h5_utils.load(FLAGS.color_components_actives % dataset)
    cc_inactives = h5_utils.load(FLAGS.color_components_inactives % dataset)
    features['color_components'] = np.concatenate((
        cc_actives['color_tanimoto'], cc_inactives['color_tanimoto'])).squeeze()

    # Tversky.
    cc_features = {}
    for feature in ['color_overlap', 'ref_self_color', 'fit_self_color']:
        cc_features[feature] = np.concatenate((cc_actives[feature],
                                               cc_inactives[feature])).squeeze()
    features['color_components_tversky'] = np.true_divide(
        cc_features['color_overlap'],
        0.95 * cc_features['ref_self_color'] +
        0.05 * cc_features['fit_self_color'])
    # If both molecules have no color atoms, color components will contain NaNs.
    mask = np.logical_and(cc_features['ref_self_color'] == 0,
                          cc_features['fit_self_color'] == 0)
    features['color_components_tversky'][mask] = 0
    assert not np.count_nonzero(np.isnan(features['color_components_tversky']))

    # Color atom overlaps.
    cao_actives = h5_utils.load(FLAGS.color_atom_overlaps_actives % dataset)
    cao_inactives = h5_utils.load(FLAGS.color_atom_overlaps_inactives % dataset)
    features['color_atom_overlaps'] = np.concatenate((
        cao_actives['color_atom_overlaps'],
        cao_inactives['color_atom_overlaps'])).squeeze()
    features['color_atom_overlaps_mask'] = np.concatenate((
        cao_actives['mask'], cao_inactives['mask'])).squeeze()

    # Sanity checks.
    for _, value in features.iteritems():
        assert value.ndim in [2, 3]
        assert value.shape[0] == labels.size

    return features, labels


def get_cv(labels):
    """Get a cross-validation iterator (NOT generator)."""
    cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True,
                                          random_state=20160416)
    return list(cv)


def get_model():
    if FLAGS.model == 'logistic':
        return linear_model.LogisticRegressionCV(class_weight='balanced',
                                                 scoring='roc_auc',
                                                 n_jobs=FLAGS.n_jobs,
                                                 max_iter=10000, verbose=1)
    elif FLAGS.model == 'random_forest':
        return ensemble.RandomForestClassifier(n_estimators=100,
                                               n_jobs=FLAGS.n_jobs,
                                               class_weight='balanced',
                                               verbose=1)
    elif FLAGS.model == 'svm':
        return grid_search.GridSearchCV(
            estimator=svm.SVC(kernel='rbf', gamma='auto',
                              class_weight='balanced'),
            param_grid={'C': np.logspace(-4, 4, 10)}, scoring='roc_auc',
            n_jobs=FLAGS.n_jobs, verbose=1)
    else:
        raise ValueError('Unrecognized model %s' % FLAGS.model)


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


def build_model(features, labels, cv, name, index=None, rocs=False):
    """Get cross-validation metrics for a single model."""
    fold_y_pred = []
    fold_y_true = []
    assert features.ndim == 2
    for fold, (train, test) in enumerate(cv):
        prefix = '%s-%s-fold-%d' % (FLAGS.prefix, name, fold)
        if index is not None:
            prefix += '-ref-%d' % index
        if rocs:
            y_pred = features[test].squeeze()
        else:
            model = get_model()
            model.fit(features[train], labels[train])
            # Save trained models.
            with gzip.open('%s-model.pkl.gz' % prefix, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            try:
                y_pred = model.predict_proba(features[test])[:, 1]
            except AttributeError:
                y_pred = model.decision_function(features[test])
        fold_y_pred.append(y_pred)
        y_true = labels[test]
        fold_y_true.append(y_true)
        # Save model output.
        assert np.array_equal(y_true.shape, y_pred.shape)
        assert y_true.ndim == 1
        with gzip.open('%s-output.pkl.gz' % prefix, 'wb') as f:
            pickle.dump(pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}), f,
                        pickle.HIGHEST_PROTOCOL)
    return get_cv_metrics(fold_y_true, fold_y_pred)


def build_models(features, labels, rows, dataset, index=None):
    """Build models using cross-validation.

    Within each fold, use LogisticRegressionCV or RandomForestClassifier on the
    training data and predict on the test set.
    """
    cv = get_cv(labels)

    # Baseline: ROCS TanimotoCombo.
    scores = build_model(features['combo_tanimoto'], labels, cv, 'rocs',
                         index=index, rocs=True)
    logging.info('ROCS TanimotoCombo: %.3f', np.mean(scores['auc']))
    add_rows('rocs', scores, rows, dataset, index)

    # Baseline: ROCS TverskyCombo.
    scores = build_model(features['combo_tversky'], labels, cv, 'rocs_tversky',
                         index=index, rocs=True)
    logging.info('ROCS TverskyCombo: %.3f', np.mean(scores['auc']))
    add_rows('rocs_tversky', scores, rows, dataset, index)

    # ROCS shape + color (Tanimoto)
    scores = build_model(np.hstack((features['shape_tanimoto'],
                                    features['color_tanimoto'])),
                         labels, cv, 'shape_color', index=index)
    logging.info('ROCS shape + color (Tanimoto): %.3f', np.mean(scores['auc']))
    add_rows('shape_color', scores, rows, dataset, index)

    # ROCS shape + color (Tversky)
    scores = build_model(np.hstack((features['shape_tversky'],
                                    features['color_tversky'])),
                         labels, cv, 'shape_color_tversky', index=index)
    logging.info('ROCS shape + color (Tversky): %.3f', np.mean(scores['auc']))
    add_rows('shape_color_tversky', scores, rows, dataset, index)

    # ROCS shape + color components (Tanimoto).
    scores = build_model(np.hstack((features['shape_tanimoto'],
                                    features['color_components'])),
                         labels, cv, 'shape_color_components', index=index)
    logging.info('ROCS shape + color components (Tanimoto): %.3f',
                 np.mean(scores['auc']))
    add_rows('shape_color_components', scores, rows, dataset, index)

    # ROCS shape + color components (Tversky).
    scores = build_model(np.hstack((features['shape_tversky'],
                                    features['color_components_tversky'])),
                         labels, cv, 'shape_color_components_tversky',
                         index=index)
    logging.info('ROCS shape + color components (Tversky): %.3f',
                 np.mean(scores['auc']))
    add_rows('shape_color_components_tversky', scores, rows, dataset, index)

    # ROCS shape + color atom overlaps
    scores = build_model(np.hstack((features['shape_tanimoto'],
                                    features['color_atom_overlaps'])),
                         labels, cv, 'shape_color_overlaps', index=index)
    logging.info('ROCS shape + color overlaps: %.3f', np.mean(scores['auc']))
    add_rows('shape_color_overlaps', scores, rows, dataset, index)

    # ROCS shape + color components + color atom overlaps (Tanimoto)
    scores = build_model(np.hstack((features['shape_tanimoto'],
                                    features['color_components'],
                                    features['color_atom_overlaps'])),
                         labels, cv, 'shape_color_components_overlaps',
                         index=index)
    logging.info('ROCS shape + color components and overlaps (Tanimoto): %.3f',
                 np.mean(scores['auc']))
    add_rows('shape_color_components_overlaps', scores, rows, dataset, index)

    # ROCS shape + color components + color atom overlaps (Tversky)
    scores = build_model(np.hstack((features['shape_tversky'],
                                    features['color_components_tversky'],
                                    features['color_atom_overlaps'])),
                         labels, cv, 'shape_color_components_tversky_overlaps',
                         index=index)
    logging.info('ROCS shape + color components and overlaps (Tversky): %.3f',
                 np.mean(scores['auc']))
    add_rows('shape_color_components_tversky_overlaps', scores, rows, dataset,
             index)


def cycle_build_models(features, labels, rows, dataset):
    """Cycle reference molecule.

    Use each active in turn as the reference molecule, removing it from the
    dataset.
    """
    actives = np.where(labels == 1)[0]
    for active_idx in actives:
        logging.info('Active index: %d', active_idx)
        keep = np.setdiff1d(np.arange(len(labels)), active_idx)
        assert len(keep) == len(labels) - 1

        # Remove reference molecule(s) from dataset.
        # Also get features specific to reference molecule(s).
        this_features = {}
        for key, value in features.iteritems():
            if key == 'color_atom_overlaps_mask':
                continue
            sel = value[keep][:, active_idx]
            if key == 'color_atom_overlaps':
                mask = features['color_atom_overlaps_mask'][keep][:, active_idx]
                sel = sel[~mask]  # Won't work for multiple reference mols?
            if sel.ndim != 2:
                sel = np.reshape(sel, (len(keep), -1))
            this_features[key] = sel
        this_labels = labels[keep]
        build_models(this_features, this_labels, rows, dataset, active_idx)


def main():
    datasets = load_datasets()
    rows = []
    for dataset in datasets:
        if dataset in FAILED and FLAGS.skip_failures:
            logging.info('Skipping %s', dataset)
            continue
        logging.info(dataset)
        features, labels = load_features_and_labels(dataset)
        if FLAGS.cycle:
            cycle_build_models(features, labels, rows, dataset)
        else:
            build_models(features, labels, rows, dataset)
    df = pd.DataFrame(rows)
    with gzip.open('%s.pkl.gz' % FLAGS.prefix, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    flags.MarkFlagAsRequired('rocs_actives')
    flags.MarkFlagAsRequired('rocs_inactives')
    flags.MarkFlagAsRequired('color_components_actives')
    flags.MarkFlagAsRequired('color_components_inactives')
    flags.MarkFlagAsRequired('color_atom_overlaps_actives')
    flags.MarkFlagAsRequired('color_atom_overlaps_inactives')
    flags.MarkFlagAsRequired('prefix')
    FLAGS(sys.argv)
    main()
