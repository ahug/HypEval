import sys
import numpy as np

PATH_TO_HYPEVAL = '../'
PATH_TO_DATA = '../data'

sys.path.append(PATH_TO_HYPEVAL)

import hypeval

if __name__ == '__main__':
    import random

    score_fcts_single = {
        'random': lambda *x: random.random()
    }
    score_fcts_batched = {
        'random': lambda x: (np.random.rand(len(x)), np.arange(len(x)))
    }

    hyp_eval = hypeval.HyponomyEvaluator(PATH_TO_DATA, append_missing=True, verbose=False)

    res = hyp_eval.evaluate(score_fcts_single['random'], 'all')
    res_by_category = hyp_eval.evaluate(score_fcts_single['random'], 'all', by_category=True)
    res_inverted = hyp_eval.evaluate(score_fcts_single['random'], 'all', inverted_pairs=True)
    res_batched = hyp_eval.evaluate(score_fcts_batched['random'], 'all', batched=True)

    import pprint
    pprint.pprint(res_by_category)
