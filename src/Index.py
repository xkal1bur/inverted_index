from typing import List, Any, Dict, DefaultDict, Tuple
from pandas.io.parsers import TextFileReader
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
import os
import json

nltk.download('stopwords')
nltk.download('punkt')


src_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(src_dir, os.pardir, 'data')
blocks_dir = os.path.join(data_dir, 'blocks')

tf_idf_json = os.path.join(data_dir, 'tf_idf.json')
doc_norms_json = os.path.join(data_dir, 'doc_norms.json')

class InvertedIndex:
    def __init__(self, file_name: str, index_col: str, block_size: int, lang: str = "english") -> None:
        self.file_name: str = file_name
        self.block_size: int = block_size
        self.stemmer = SnowballStemmer(lang)
        self.stoplist: List[str] = stopwords.words(lang)

        #self.n_blocks: int = 0 # tmp
        self.n_blocks: int = 5 # tmp
        self.block_size_list: List[int] = []
        self.build_index(index_col, lang)

    
    def build_index(self, index_col: str, lang: str) -> None:
        read_path = os.path.join(data_dir, self.file_name)
        chunks: TextFileReader = pd.read_csv(read_path, chunksize=self.block_size, usecols=[index_col, 'language'])
        os.makedirs(os.path.join(data_dir, 'blocks'), exist_ok=True)
        for i, chunk in enumerate(chunks, start=1):
            BM = (chunk['language'] == lang[:2]) & (pd.notna(chunk[index_col]))
            partial_index: DefaultDict[str, Counter] = defaultdict(Counter)
            for idx, doc in chunk[BM].iterrows():
                try:
                    partial_counter = self.preprocess(doc.iloc[0])
                except Exception as e:
                    print(f"Error at {idx = },{i = }")
                for word, tf in partial_counter.items():
                    partial_index[word][idx] = tf
            
            ordered_partial_index: Dict[str, Dict[str, int]] = {}

            self.block_size_list.append(0)
            for term, posting in sorted(partial_index.items()):
                ordered_partial_index[term] = dict(sorted(posting.items()))
                self.block_size_list[i-1] += len(posting)
                

            self.write_block(ordered_partial_index, block_num=i, height=0)
            
        with open(os.path.join(data_dir, 'block_sizes.json'), 'w', encoding='utf-8') as file: 
            json.dump(self.block_size_list, file, indent=4) 
        print(f"Blocks created: {i = }")
        self.n_blocks = i
        self.merge_blocks()


    def merge_blocks(self) -> None:
        #self.block_size_list = json.load(open(os.path.join(data_dir, 'block_sizes.json'), 'r', encoding='utf-8'))
        # esa carga no debería estar
        print(f"{self.n_blocks = }")
        self.mergesort(1, self.n_blocks)
    
    def mergesort(self, p:int, r:int) -> int:
        if p >= r:
            return -1
        q = (p+r)//2
        print(f"MERGESORT: {p=} {r=} {q=}")
        left_h: int = self.mergesort(p, q)
        right_h: int = self.mergesort(q+1, r)
        # print(f"{p=} {r=} {q=}")
        h: int = max(left_h, right_h) + 1
        self.merge(p,q,r,h)
        return h
    
    def merge(self, p:int, q:int, r:int, height:int) -> None:
        nl: int = q - p + 1
        nr: int = r - q

        i_ext: int = 0
        j_ext: int = 0

        print(f"{str(height).center(100, '=')}")
        print(f"{height}")
        print(f"nl: {nl}  nr: {nr}")
        print(f"{i_ext=},{nl=},  {j_ext=}, {nr=}")
        while i_ext < nl and j_ext < nr:
            print(f"{q + j_ext + 1 = }".center(28, '-'))
            d1, blockl_k, blockl_v, sizel = self.load_block(block_num=p + i_ext, height=height)
            d2, blockr_k, blockr_v, sizer = self.load_block(block_num=q + j_ext, height=height)
            new_block: Dict[str, Dict[str, int]] = self.merge_dicts(d1, d2)

            block1, block2 = self.split_dict(new_block, (sizel + sizer)//2)
            
            print(f"block num 1: {p + i_ext}  block num 2: {p + i_ext + j_ext + 1}")
            self.write_block(block1, p + i_ext, height+1)
            self.write_block(block2, q + j_ext, height+1)
            i_ext += 1
            j_ext += 1
        while i_ext < nl:
            print("EXTENDIENDO I".center(100, '-'))
            d1, blockl_k, blockl_v, sizel = self.load_block(p + i_ext, height)
            self.write_block(d1, p + i_ext, height+1)
            i_ext += 1
        while j_ext < nr:
            print("EXTENDIENDO J".center(100, '-'))
            d2, blockr_k, blockr_v, sizer = self.load_block(q + j_ext, height)
            self.write_block(d2, q + j_ext, height+1)
            j_ext += 1
        
        
    def merge_dicts(self, d1: Dict[str, Dict[str, int]], d2: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        merged: Dict[str, Dict[str, int]] = {}
        keys1: List[str] = list(d1.keys())
        keys2: List[str] = list(d2.keys())

        i, j = 0, 0
        while i < len(keys1) and j < len(keys2):
            if keys1[i] < keys2[j]:
                merged[keys1[i]] = d1[keys1[i]]
                i += 1
            elif keys1[i] > keys2[j]:
                merged[keys2[j]] = d2[keys2[j]]
                j += 1
            else:  # keys1[i] == keys2[j]
                merged[keys1[i]] = {}
                nested_keys1: List[str] = list(d1[keys1[i]].keys())
                nested_keys2: List[str] = list(d2[keys1[i]].keys())

                k, l = 0, 0
                while k < len(nested_keys1) and l < len(nested_keys2):
                    if nested_keys1[k] < nested_keys2[l]:
                        merged[keys1[i]][nested_keys1[k]] = d1[keys1[i]][nested_keys1[k]]
                        k += 1
                    elif nested_keys1[k] > nested_keys2[l]:
                        merged[keys1[i]][nested_keys2[l]] = d2[keys1[i]][nested_keys2[l]]
                        l += 1
                    else:  # nested_keys1[k] == nested_keys2[l]
                        merged[keys1[i]][nested_keys1[k]] = d2[keys1[i]][nested_keys2[l]]
                        k += 1
                        l += 1

                while k < len(nested_keys1):
                    merged[keys1[i]][nested_keys1[k]] = d1[keys1[i]][nested_keys1[k]]
                    k += 1

                while l < len(nested_keys2):
                    merged[keys1[i]][nested_keys2[l]] = d2[keys1[i]][nested_keys2[l]]
                    l += 1

                i += 1
                j += 1

        while i < len(keys1):
            merged[keys1[i]] = d1[keys1[i]]
            i += 1

        while j < len(keys2):
            merged[keys2[j]] = d2[keys2[j]]
            j += 1

        return merged
    def split_dict(self, input_dict: Dict[str, Dict[str, int]], split_size: int) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        dict1 = {}
        dict2 = {}
        current_size = 0

        for key, value in input_dict.items():
            if current_size < split_size:
                dict1[key] = value
                current_size += len(value)
            else:
                dict2[key] = value
        return dict1, dict2
    def write_block(self, block: Dict[str, Dict[str, int]], block_num: int, height: int) -> None:
        print(f"Writing block {block_num} height {height}")
        block_path = os.path.join(blocks_dir, f"block_{block_num}__height_{height}.json")
        with open(block_path, 'w', encoding='utf-8') as file:
            json.dump(block, file, indent=4)

    def load_block(self, block_num: int, height:int = 0) -> Tuple[Dict[str, Dict[str, int]], List[str], List[Dict[str, int]], int]:
        print(f"Loading block {block_num} height {height}")
        block_path = os.path.join(blocks_dir, f"block_{block_num}__height_{height}.json")
        with open(block_path, 'r', encoding='utf-8') as file:
            block_index: Dict[str, Dict[str, int]] =  json.load(file)
            docsizes: int = sum(len(posting) for posting in block_index.values())
            keys: List[str] = list(block_index.keys())
            values: List[Dict[str, int]] = list(block_index.values())
        return block_index, keys, values, docsizes

    def preprocess(self, doc: str) -> Dict[str, int]:
        words: List[str] = word_tokenize(doc.lower())
        words_f : List[str] = [w for w in words if w not in self.stoplist and w.isalpha()]
        return Counter([self.stemmer.stem(w) for w in words_f])

if __name__ == "__main__":
    xd: InvertedIndex = InvertedIndex("spotify_songs.csv", "lyrics", 4000)

