import nltk
import string
from typing import Dict, List, Tuple, Union, Callable, Optional
import numpy as np
import torch
import json
import faiss
from langdetect import detect
from models.knrm import KNRM


def check_eng_language(text: str) -> bool:
    return detect(text) == 'en'


class Handler:
    def __init__(self,
                 path_emb_knrm: str,
                 path_mlp: str,
                 path_vocab: str,
                 num_suggestions: int = 10,
                 num_candidates: int = 20,
                 max_len: int = 30
                 ):
        self.documents = None
        self._model_is_ready = False

        self.path_emb_knrm = path_emb_knrm
        self.path_mlp = path_mlp
        self.path_vocab = path_vocab

        self.num_suggestions = num_suggestions
        self.num_candidates = num_candidates

        self.max_len = max_len

        self.model, self.vocab = self._build_knrm_model()

        self.emb_weights = self.model.embeddings.state_dict()['weight'].numpy()

        self.index = None
        self._index_size = None

    @property
    def model_is_ready(self):
        return self._model_is_ready

    @property
    def index_is_ready(self):
        return self._index_size is not None and self._index_size > 0

    @property
    def index_size(self):
        return self._index_size

    def _build_knrm_model(self) -> Tuple[KNRM, Dict[str, int]]:
        embeddings_weights = torch.load(self.path_emb_knrm)['weight']
        mlp_weights = torch.load(self.path_mlp)

        with open(self.path_vocab, 'r') as f:
            vocab = json.load(f)

        model = KNRM(embedding_matrix=embeddings_weights,
                     freeze_embeddings=True,
                     mlp_weights=mlp_weights)

        self._model_is_ready = True

        return model, vocab

    def simple_preproc(self, inp_str: str) -> List[str]:

        for p in string.punctuation:
            inp_str = inp_str.replace(p, " ")

        return nltk.word_tokenize(inp_str.lower().strip())

    def _vectorize_sentence(self, sentence: str) -> np.array:

        tokens = self.simple_preproc(sentence)

        sentence_vectors = [self.emb_weights[self.vocab.get(t, self.vocab['OOV'])] for t in tokens]

        if len(sentence_vectors) == 0:
            sentence_vectors = [self.emb_weights[self.vocab['OOV']]]

        return np.array(np.mean(sentence_vectors, axis=0))

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(token, self.vocab['OOV']) for token in tokenized_text[:self.max_len]]

    def _convert_text_to_token_idxs(self, text: str) -> List[int]:
        return self._tokenized_text_to_index(self.simple_preproc(text))

    def build_index(self, documents: Dict[str, str]):
        self.documents = documents

        sentence_vectors = []
        ids = []

        for id in self.documents.keys():
            sentence_vectors.append(self._vectorize_sentence(self.documents[id]))
            ids.append(int(id))

        embeddings = np.array([embedding for embedding in sentence_vectors]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(ids))

        self._index_size = self.index.ntotal

    def _rank_suggestions(self, query: str, suggestions: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

        scores = []

        for suggestion in suggestions:
            inputs = {'query': torch.Tensor(self._convert_text_to_token_idxs(query)).to(torch.int64),
                      'document': torch.Tensor(self._convert_text_to_token_idxs(suggestion[1])).to(torch.int64),
                      }

            scores.append(self.model.predict(inputs).detach().numpy()[0][0])

        res_ids = (-np.array(scores)).argsort()[:self.num_suggestions]
        res = [suggestions[i] for i in res_ids.tolist()]

        return res

    def similarity_search(self, queries: Dict[str, List[str]]) -> Tuple[
        List[bool], List[Optional[List[Tuple[str, str]]]]]:

        lang_check_list = []
        suggestions = []

        for query in queries:
            is_eng = check_eng_language(query)
            lang_check_list.append(is_eng)

            if not is_eng:
                suggestions.append(None)

            else:
                query_vector = self._vectorize_sentence(query).reshape(1, -1)

                _, suggestion_ids = self.index.search(np.array(query_vector.astype('float32')),
                                                      self.num_candidates)
                cur_suggestions = [(str(id), self.documents.get(str(id), '0')) for id in suggestion_ids[0] if id != -1]

                suggestions.append(self._rank_suggestions(query, cur_suggestions))

        return lang_check_list, suggestions
