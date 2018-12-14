import numpy as np

def score_fct_vocab(single_score_fct, vocab):
    def score_fct(pair):
        if pair[0] not in vocab or pair[1] not in vocab:
            return None
        return single_score_fct(pair)

    return score_fct


def score_fct_vector_dict(single_score_fct, vector_dict):
    def score_fct(hypo, hyper):
        if hypo not in vector_dict or hyper not in vector_dict:
            return None
        return single_score_fct(vector_dict[hypo], vector_dict[hyper])

    return score_fct


def score_fct_vector_dict_batched(batch_score_fct, vector_dict):
    def score_fct(pairs):
        found_ix = []
        hypo_vecs, hyper_vecs = [], []
        for i, pair in enumerate(pairs):
            if pair[0] not in vector_dict or pair[1] not in vector_dict:
                continue
            found_ix.append(i)

            hypo_vecs.append(vector_dict[pair[0]])
            hyper_vecs.append(vector_dict[pair[1]])
        hypo_vecs, hyper_vecs = np.stack(hypo_vecs), np.stack(hyper_vecs)
        return batch_score_fct(hypo_vecs, hyper_vecs), np.array(found_ix)

    return score_fct