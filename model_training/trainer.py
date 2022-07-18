import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import os
import json


class Trainer:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 path_to_save_weights: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [10, 5],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10,
                 ):

        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.path_to_save_weights = path_to_save_weights

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
                                           self.idx_to_text_mapping_dev,
                                           vocab=self.vocab, oov_val=self.vocab['OOV'],
                                           preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)


    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        for p in string.punctuation:
            inp_str = inp_str.replace(p, " ")

        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        return nltk.word_tokenize(self.hadle_punctuation(inp_str).lower().strip())

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {word: count for word, count in vocab.items() if count >= min_occurancies}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        all_texts = []

        for df in list_of_df:
            all_texts += list(df['text_left'].values)
            all_texts += list(df['text_right'].values)

        word2count = Counter(self.simple_preproc(' '.join(set(all_texts))))

        # word filtering
        filtered_word2count = self._filter_rare_words(word2count, min_occurancies)

        return list(filtered_word2count.keys())

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:

        file = open(file_path)

        embeddings = {}

        for line in file:
            line = line.split(' ')
            word = line[0]
            embedding = line[1:]
            embeddings[word] = embedding
        file.close()

        return embeddings

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:

        np.random.seed(random_seed)

        word2embedding = self._read_glove_embeddings(file_path)
        unk_words = list(set(inner_keys).difference(set(word2embedding.keys())))
        known_words = list(set(inner_keys).intersection(set(word2embedding.keys())))

        embedding_size = len(word2embedding[known_words[0]])
        unk_embedding = np.random.uniform(-rand_uni_bound, rand_uni_bound, embedding_size)

        embeddings = np.zeros((len(inner_keys) + 2, embedding_size))
        word2index = {"PAD": 0, "OOV": 1}
        unk_words += ["PAD", "OOV"]

        embeddings[1, :] = unk_embedding

        for index, word in enumerate(inner_keys, 2):
            embeddings[index, :] = word2embedding.get(word, unk_embedding)
            word2index[word] = index

        return embeddings, word2index, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int) -> List[List[Union[str, float]]]:
        fill_top_to, min_group_size = -1, 2

        inp_df = inp_df[['id_left', 'id_right', 'label']]
        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        group_size = inp_df.groupby('id_left').size()
        left_ind_to_use = group_size[group_size >= min_group_size].index.tolist()
        groups = inp_df[inp_df['id_left'].isin(left_ind_to_use)].groupby('id_left')

        np.random.seed(seed)

        out_pairs = []
        for id_left, group in groups:
            ones_ids = group[group.label == 1].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union(set(id_left))
                pad_sample = np.random.choice(list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for doc1, doc2 in [(x, y) for x in ones_ids for y in zeroes_ids]:
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])
            for doc1, doc2 in [(x, y) for x in ones_ids for y in pad_sample]:
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])
            for doc1, doc2 in [(x, y) for x in zeroes_ids for y in pad_sample]:
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])

        return out_pairs

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        # допишите ваш код здесь  (обратите внимание, что используются вектора numpy)

        current_dcg = self._dcg_k(torch.Tensor(ys_true), torch.Tensor(ys_pred), ndcg_top_k)
        ideal_dcg = self._dcg_k(torch.Tensor(ys_true), torch.Tensor(ys_true), ndcg_top_k)
        return current_dcg / ideal_dcg

    def compute_gain(self, y_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == "const":
            return y_value
        elif gain_scheme == "exp2":
            return 2 ** y_value - 1

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        sorted_true = ys_true[indices][:k].numpy()
        gain = self.compute_gain(sorted_true)
        discount = [math.log2(float(x)) for x in range(2, len(sorted_true) + 2)]
        discounted_gain = float((gain / discount).sum())
        return discounted_gain


    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()

        best_ndcg = 0

        for epoch in range(n_epochs):

            if epoch % self.change_train_loader_ep == 0:
                cur_subset = self.sample_data_for_train_iter(self.glue_train_df, epoch)
                train_dataset = TrainTripletsDataset(cur_subset, self.idx_to_text_mapping_train,
                                                     self.vocab, oov_val=self.vocab["OOV"],
                                                     preproc_func=self.simple_preproc
                                                     )
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.dataloader_bs, num_workers=0,
                    collate_fn=collate_fn, shuffle=True)

            for batch in train_dataloader:
                inp_1, inp_2, labels = batch
                preds = self.model(inp_1, inp_2)

                loss = criterion(preds, labels)
                # opt.zero_grad()
                loss.backward()
                opt.step()

            # if epoch > 5:
            ndcg = self.valid(self.model, self.val_dataloader)

            print(f'{epoch} epoch; \t valid ndcg: {ndcg}')

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                self._save_model(path_dir=self.path_to_save_weights)

    def _save_model(self, path_dir):
        os.makedirs(path_dir, exist_ok=True)

        torch.save(self.model.embeddings.state_dict(), os.path.join(path_dir, 'embeddings.bin'))
        torch.save(self.model.mlp.state_dict(), os.path.join(path_dir, 'mlp.bin'))

        with open(os.path.join(path_dir, 'vocab.json'), "w") as f:
            json.dump(self.vocab, f, indent=4)


