from __future__ import unicode_literals
import re
import unicodedata
import tarfile
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer



class Hougen:

    def unicode_normalize(self,cls, s):
        pt = re.compile('([{}]+)'.format(cls))
        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c
        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('－', '-', s)
        return s

    def remove_extra_spaces(self, s):
        s = re.sub('[ 　]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                        '\u3040-\u309F',  # HIRAGANA
                        '\u30A0-\u30FF',  # KATAKANA
                        '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                        '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                        ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s

    def normalize_neologd(self,s):
        s = s.strip()
        s = self.unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = self.remove_extra_spaces(s)
        s = self.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s

    def remove_brackets(self,text):
        text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
        return text

    def normalize_text(self,text):
        assert "\n" not in text and "\r" not in text
        text = text.replace("\t", " ")
        text = text.strip()
        text = self.normalize_neologd(text)
        text = text.lower()
        return text

    def read_title_body(self,file):
        next(file)
        next(file)
        title = next(file).decode("utf-8").strip()
        title = self.normalize_text(self.remove_brackets(title))
        body = self.normalize_text(" ".join([line.decode("utf-8").strip() for line in file.readlines()]))
        return title, body

    def hougen(self,texts):
        
        model_name_or_path="sonoisa/t5-base-japanese",
        MODEL_DIR = "model"
        
        # 各種ハイパーパラメータ
        args_dict = dict(
            # data_dir="/content/data",  # データセットのディレクトリ
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=model_name_or_path,
            
            learning_rate=3e-4,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            warmup_steps=0,
            gradient_accumulation_steps=1,

            # n_gpu=1 if USE_GPU else 0,
            early_stop_callback=False,
            fp_16=False,
            max_grad_norm=1.0,
            seed=42,
        )

        # 学習に用いるハイパーパラメータを設定する
        args_dict.update({
            "max_input_length":  64,  # 入力文の最大トークン数
            "max_target_length": 64,  # 出力文の最大トークン数
            "train_batch_size":  64,  # 訓練時のバッチサイズ
            "eval_batch_size":   64,  # テスト時のバッチサイズ
            "num_train_epochs":  20,  # 訓練するエポック数
            })
        args = argparse.Namespace(**args_dict)

        # GPU利用有無
        # USE_GPU = torch.cuda.is_available()

        # トークナイザー（SentencePiece）
        tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)
        # 学習済みモデル
        trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

        body = texts

        MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
        MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

        def preprocess_body(text):
            return self.normalize_text(text.replace("\n", " "))

        # 推論モード設定
        trained_model.eval()

        # 前処理とトークナイズを行う
        inputs = [preprocess_body(body)]
        batch = tokenizer.batch_encode_plus(
            inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
            padding="longest", return_tensors="pt")

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        # if USE_GPU:
        #     input_ids = input_ids.cuda()
        #     input_mask = input_mask.cuda()

        # 生成処理を行う
        outputs = trained_model.generate(
            input_ids=input_ids, attention_mask=input_mask, 
            max_length=MAX_TARGET_LENGTH,
            temperature=2.0,          # 生成にランダム性を入れる温度パラメータ
            num_beams=10,             # ビームサーチの探索幅
            diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ
            num_beam_groups=10,       # ビームサーチのグループ数
            num_return_sequences=1,  # 生成する文の数
            repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

        # 生成されたトークン列を文字列に変換する
        generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False) 
                            for ids in outputs]

        # 生成されたタイトルを表示する
        # print('変換前： ', body)
        # for i, title in enumerate(generated_titles):
        #     print('変換後： ',f"{i+1:2}. {title}")

        return generated_titles[0]