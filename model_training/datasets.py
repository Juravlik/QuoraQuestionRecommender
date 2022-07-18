from typing import Dict, List, Tuple, Union, Callable
import torch


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(token, self.oov_val) for token in tokenized_text[:self.max_len]]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        return self._tokenized_text_to_index(self.preproc_func(self.idx_to_text_mapping[str(idx)]))

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        query_idx, doc_left_idx, doc_right_idx, label = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(query_idx)
        doc_left_tokens = self._convert_text_idx_to_token_idxs(doc_left_idx)
        doc_right_tokens = self._convert_text_idx_to_token_idxs(doc_right_idx)

        output_1 = {
            "query": query_tokens,
            "document": doc_left_tokens
        }

        output_2 = {
            "query": query_tokens,
            "document": doc_right_tokens
        }

        return output_1, output_2, label


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        query_idx, doc_left_idx, label = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(query_idx)
        doc_left_tokens = self._convert_text_idx_to_token_idxs(doc_left_idx)

        output_1 = {
            "query": query_tokens,
            "document": doc_left_tokens
        }

        return output_1, label


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels