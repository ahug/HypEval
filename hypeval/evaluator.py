import os
import numpy as np
import logging

from collections import defaultdict
from sklearn.metrics import average_precision_score

import time


class HyponomyEvaluator():
    def __init__(self, dataset_dir, append_missing=True, verbose=False):
        """Provides a simple interface to evaluate models on common hypernymy detection datasets

        Parameters
        ----------
        dataset_dir: string
            Directory containing the hypernymy dataset files.

        append_missing: bool
            Whether the missing pairs (OOV) should be appended at the end of the list, essentially assuming that
            they are not in a hypernymy relation. This procedure is followed e.g. by
            (Distributional Inclusion Vector Embedding for Unsupervised Hypernymy Detection, Chang et al.)

        verbose: bool
            Print more detailed information.

        """
        self.dataset_dir = dataset_dir
        self.append_missing = append_missing
        self.verbose = verbose

        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('hyp-eval')


    def read_dataset(self, dataset_path, inverted_pairs=False):
        """Reads the hypernymy pairs, relation type and the true label from the given file and returns these
           four properties a separate lists.

        Parameters
        __________
        dataset_path: string
            Path of the dataset file. The file should contain one positive/negative pair per line. The format of each
            line should be of the following form:
                hyponym  hypernym    label   relation-type
            each separated by a tab.

        inverted_pairs: bool
            Whether only the positive pairs + all positive pairs inverted (switch hyponym <-> hypernym in positive
            pairs) should be returned. This can be helpful to check how well a model can the directionality of the
            hypernymy relation.

        Returns
        _______
        tuple:
            relations: np.array, pairs: list[(hyponym, hypernym)], labels: np.array(dtype=bool)
        """
        with open(dataset_path) as f:
            dataset = [tuple(line.strip().split("\t")) for line in f]

            for i in range(len(dataset)):
                if len(dataset[i]) < 4:
                    raise ValueError('Encountered invalid line in "%s" on line %d: %s' % (dataset_path, i, dataset[i]))

            w1, w2, labels, relations = zip(*dataset)
            pairs = list(zip(w1, w2))
            labels = (np.array(labels) == "True")

            if inverted_pairs:
                pos_pairs = [pairs[ix] for ix, lbl in enumerate(labels) if lbl]
                neg_pairs = [(p2, p1) for p1, p2 in pos_pairs]
                pairs = pos_pairs + neg_pairs
                labels = np.array([True] * len(pos_pairs) + [False] * len(neg_pairs))
                relations = ['hyper'] * len(pos_pairs) + ['inverted'] * len(neg_pairs)

        return np.array(relations), pairs, labels


    def evaluate(self, score_fct, datasets, at_k=None, batched=False, inverted_pairs=False, by_category=False,
                 print_progress=False, debug=False):
        """Returns various evaluation metrics for the passed hypernymy datasets.

        Parameters
        __________
        score_fct: function
            A function f(x, y) that takes two words (strings) as an input and returns a score float
            or None if the score cannot be computed for the given pair (e.g. if the words are not in the vocabulary).
            The score that is returned by 'score_fct' should be high for positive pairs and smaller for negative pairs
            as the scores are sorted in descending order.

            If 'batched' is set to True, 'x' and 'y' are passed as a list of pairs (tuples). In that case, 'score_fct'
            should return an np.array and an np.array(dtype=np.int) indicating which pairs were present in the
            vocabulary.

        datasets: list[string] or string
            A string or a list of dataset names that should be used for the evaluation.

        at_k: integer
            Also computes the average precision for the top 'at_k' elements (AP@at_k).

        batched: bool
            If the scores should be computed in a batched manner. See 'score_fct' for more details.

        inverted_pairs: bool
            Whether only the positive pairs + all positive pairs inverted (switch hyponym <-> hypernym in positive
            pairs) should be returned. This can be helpful to check how well a model can the directionality of the
            hypernymy relation.

        by_category: bool
            If 'True', computes the average precision for predicting hypernymy against each of the categories
            that are present in the dataset.

        Returns
        _______
        result: dict
            Returns a dictionary which contains the AP scores for all datasets.
        """
        if isinstance(datasets, str):
            if datasets == 'all':
                import glob
                datasets = glob.glob(os.path.join(self.dataset_dir, '*.all'))
                if self.verbose:
                    datasets_repr = [ds.split("/")[-1] if "/" in ds else ds for ds in datasets]
                    self.logger.info("Using all datasets: %s" % str(datasets_repr))
            elif "," in datasets:
                datasets = datasets.split(",")
            else:
                datasets = [datasets]

        results = {}
        for dataset_name in datasets:
            start_time = time.time()

            res_dict = defaultdict(dict)

            dataset_file = os.path.join(self.dataset_dir, dataset_name)
            all_relations, all_pairs, all_labels = self.read_dataset(dataset_file, inverted_pairs=inverted_pairs)
            scores = np.empty(len(all_pairs))
            if not batched:
                for i, p in enumerate(all_pairs):
                    scores[i] = score_fct(*p)

                    if print_progress and i % 100 == 0:
                        print("Progress: %d%%" % (100. * i / len(all_pairs)))
                found_ix = np.argwhere(~np.isnan(scores)).squeeze()  # all missing entries should be None
                scores, labels, relations = scores[found_ix], all_labels[found_ix], all_relations[found_ix]
            else:
                scores, found_ix = score_fct(all_pairs)
                labels, relations = all_labels[found_ix], all_relations[found_ix]

            if self.append_missing and not inverted_pairs:  # appending makes no sense for inverted pairs
                # append the missing ones at the end
                mask = np.ones(len(all_pairs), dtype=bool)
                mask[found_ix] = False
                missing_labels = all_labels[mask]

                missing_relations = all_relations[mask]
                if len(scores) == 0:
                    raise ValueError(
                        "No scores were computed by 'score_fct'. Does 'score_fct' correctly return a score?")
                pairs_scores_zip = [(all_pairs[ix], scores[i], all_labels[ix]) for i, ix in enumerate(found_ix)]

                dummy_scores = np.min(scores) * np.ones(len(missing_labels)) - 1  # append to the end of list
                labels = np.concatenate((labels, missing_labels))
                scores = np.concatenate((scores, dummy_scores))
                relations = np.concatenate((relations, missing_relations))

                if self.verbose:
                    self.logger.info('%s: %d pairs out of %d are missing (OOV). %d/%d of these pairs are positives' % (
                        dataset_name, len(missing_labels), len(all_labels), missing_labels.sum(), len(missing_labels)))

            if self.verbose:
                dataset_name = dataset_name.split("/")[-1]
                self.logger.info("%s: %d pairs out of %d are used for evaluation (%d%%)" % (
                    dataset_name, len(scores), len(all_pairs), len(scores) / len(all_pairs) * 100))

            # hyponomy vs all other relations
            avg_prec_all = average_precision_score(labels, scores)
            res_dict["AP@all"] = avg_prec_all

            if at_k:
                scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0])[::-1])
                avg_prec_all_top_k = average_precision_score(labels[:at_k], scores[:at_k])
                res_dict["AP@%d" % at_k] = avg_prec_all_top_k

            if by_category:
                for category in np.unique(relations):
                    if category == "hyper":
                        continue
                    ix = np.logical_or(relations == "hyper", relations == category)
                    scores_by_category, labels_by_category = scores[ix], labels[ix]

                    res_dict[category]["AP@all"] = average_precision_score(labels_by_category, scores_by_category)
                    if at_k and len(labels_by_category) > at_k:
                        res_dict[category]["AP@%d" % at_k] = average_precision_score(labels_by_category[:at_k],
                                                                                     scores_by_category[:at_k])
            results[dataset_name] = res_dict

            if self.verbose:
                stop_time = time.time()
                self.logger.info("Evaluation on %s took %.2f seconds." % (dataset_name, stop_time - start_time))

        if debug:
            return results, pairs_scores_zip

        return results


    def evaluate_directionality(self, score_fct, datasets, print_progress=False):
        if isinstance(datasets, str):
            if datasets == 'all':
                import glob
                datasets = glob.glob(os.path.join(self.dataset_dir, '*.all'))
                if self.verbose:
                    datasets_repr = [ds.split("/")[-1] if "/" in ds else ds for ds in datasets]
                    self.logger.info("Using all datasets: %s" % str(datasets_repr))
            else:
                datasets = [datasets]

        results = {}
        for dataset_name in datasets:
            start_time = time.time()

            dataset_file = os.path.join(self.dataset_dir, dataset_name)
            all_relations, all_pairs, all_labels = self.read_dataset(dataset_file)
            hyper_pairs = [all_pairs[i] for i in np.argwhere(all_relations == "hyper").squeeze()]

            if self.verbose:
                self.logger.info(
                    "Found %d positive hyper pairs (%d/%d)." % (len(hyper_pairs), len(hyper_pairs), len(all_relations)))

            num_correct, num_missing = 0, 0
            for hypo, hyper in hyper_pairs:
                true_dir_score, wrong_dir_score = score_fct(hypo, hyper), score_fct(hyper, hypo)
                if true_dir_score is None or wrong_dir_score is None:
                    num_missing += 1
                    continue

                if true_dir_score > wrong_dir_score:
                    num_correct += 1

            if self.verbose:
                self.logger.info("%d pairs out of %d were missing in the vocabulary." % (num_missing, len(hyper_pairs)))
                stop_time = time.time()
                self.logger.info("Evaluation on %s took %.2f seconds." % (dataset_name, stop_time - start_time))

            accuracy = num_correct / len(hyper_pairs)
            results[dataset_name] = accuracy
        return results

