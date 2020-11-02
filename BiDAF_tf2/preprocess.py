import os
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm

import data_io as pio
from config import dat_charset_fname, dat_wordset_fname, max_char_len


class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 30
        self.max_qlen = 20
        self.max_char_len = max_char_len
        self.stride = stride
        # 限制最小频率
        self.min_count = 50
        self.charset = set()
        self.wordset = set()
        self.build_charset()

    def build_charset(self):
        for fp in self.datasets_fp:
            charset, wordset = self.dataset_info(fp)
            self.charset |= charset
            self.wordset |= wordset

        # char
        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))

        self.char2id = dict(zip(self.charset, idx))
        self.id2char = dict(zip(idx, self.charset))

        # word
        self.wordset = sorted(list(self.wordset))
        self.wordset = ['[PAD]', '[CLS]', '[SEP]'] + self.wordset + ['[UNK]']
        idx = list(range(len(self.wordset)))

        self.word2id = dict(zip(self.wordset, idx))
        self.id2word = dict(zip(idx, self.wordset))

    def dataset_info(self, inn):
        if os.path.exists(dat_charset_fname) and os.path.exists(dat_wordset_fname):
            charset = pickle.load(open(dat_charset_fname, 'rb'))
            wordset = pickle.load(open(dat_wordset_fname, 'rb'))
        else:
            dataset = pio.load(inn)

            char_content = ''
            word_content = ''
            for _, context, question, answer, _ in tqdm(self.iter_cqa(dataset)):
                char_content += context + question + answer
                word_content += ' ' + context + ' ' + question + ' ' + answer

            char_count = Counter(char_content)
            word_count = Counter(word_content.split(' '))

            # 过滤词频，返回set集合
            charset = set([char for char, count in char_count.items() if count >= self.min_count])
            wordset = set([word for word, count in word_count.items() if count >= self.min_count])

        pickle.dump(charset, open(dat_charset_fname, 'wb'))
        pickle.dump(wordset, open(dat_wordset_fname, 'wb'))
        return charset, wordset

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_word_encode = self.convert_word2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_word_encode)
        context_word_encode = self.convert_word2id(context, maxlen=left_length, end=True)
        cq_word_encode = question_word_encode + context_word_encode

        assert len(cq_word_encode) == self.max_length

        question_char_encode = self.convert_char2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_char_encode)
        context_char_encode = self.convert_char2id(context, maxlen=left_length, end=True)
        cq_char_encode = question_char_encode + context_char_encode

        return cq_word_encode, cq_char_encode

    def convert_word2id(self, sent, maxlen=None, begin=False, end=False):
        words = sent.split()
        words = ['[CLS]'] * begin + words

        if maxlen is not None:
            words = words[:maxlen - 1 * end]
            words += ['[SEP]'] * end
            words += ['[PAD]'] * (maxlen - len(words))
        else:
            words += ['[SEP]'] * end

        ids = list(map(self.get_word_id, words))

        return ids

    def convert_char2id(self, sent, maxlen=None, begin=False, end=False):
        words = sent.split()
        words = [list(word)[:self.max_char_len] if len(word) > self.max_char_len \
                     else list(word) + ['[PAD]'] * (self.max_char_len - len(word)) for word in words]

        # pad chars
        words += [['[PAD]'] * self.max_char_len] * (maxlen - len(words))

        words += [['[CLS]'] + ['[PAD]'] * (self.max_char_len - 1)] * begin + words

        if maxlen is not None:
            words = words[:maxlen - 1 * end]
            words += [['[SEP]'] + ['[PAD]'] * (self.max_char_len - 1)] * end
            words += [['[PAD]'] * self.max_char_len] * (maxlen - len(words))
        else:
            words += [['[SEP]'] + ['[PAD]'] * (self.max_char_len - 1)] * end

        ids = [self.get_char_id(word) for word in words]

        return ids

    def get_char_id(self, word):
        return [self.char2id.get(char, self.char2id['[UNK]']) for char in word]

    def get_word_id(self, word):
        return self.get_id(word)

    def get_id(self, word):
        # 获取index,无则填充UNK
        return self.word2id.get(word, self.word2id['[UNK]'])

    def get_dataset(self, ds_fp, pickle_data_path):
        if pickle_data_path and os.path.exists(pickle_data_path):
            print('loading pickle_dataset:', pickle_data_path)
            proc_data = pickle.load(open(pickle_data_path, 'rb'))
        else:
            contents_word_ids, questions_word_ids, begin_and_end = [], [], []
            # ADD char list
            contents_char_ids, questions_char_ids = [], []
            for _, content_word_ids, question_word_ids, content_char_ids, question_char_ids, begin, end in tqdm(
                    self.get_data(
                        ds_fp)):
                # WORD
                contents_word_ids.append(content_word_ids)
                questions_word_ids.append(question_word_ids)
                # CHAR
                contents_char_ids.append(content_char_ids)
                questions_char_ids.append(question_char_ids)
                begin_and_end.append((begin, end))

            contents_word_ids = np.array(contents_word_ids, dtype=np.int32)
            questions_word_ids = np.array(questions_word_ids, dtype=np.int32)
            contents_char_ids = np.array(contents_char_ids, dtype=np.int32)
            questions_char_ids = np.array(questions_char_ids, dtype=np.int32)
            begin_and_end = np.array(begin_and_end, dtype=np.int32)

            proc_data = [contents_word_ids, questions_word_ids, contents_char_ids, questions_char_ids,
                         begin_and_end]
            pickle.dump(proc_data, open(pickle_data_path, 'wb'))
            print('save proc_data done', pickle_data_path)
        return proc_data

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            content_word_ids = self.get_sent_word_ids(context, self.max_clen)
            question_word_ids = self.get_sent_word_ids(question, self.max_qlen)

            content_char_ids = self.get_sent_char_ids(context, self.max_clen)
            question_char_ids = self.get_sent_char_ids(question, self.max_qlen)
            begin, end = answer_start, answer_start + len(text)
            if end >= len(content_word_ids):
                begin = end = 0
            yield qid, content_word_ids, question_word_ids, content_char_ids, question_char_ids, begin, end

    def get_sent_word_ids(self, sent, maxlen):
        return self.convert_word2id(sent, maxlen=maxlen, end=True)

    def get_sent_char_ids(self, sent, maxlen):
        return self.convert_char2id(sent, maxlen=maxlen, end=True)


if __name__ == '__main__':
    p = Preprocessor([
        # './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        # './data/squad/dev-v1.1.json'
    ])
    print(len(p.word2id))
    print(len(p.char2id))
    # [['T', 'o'], ['w', 'h', 'o', 'm'], ['d', 'i', 'd'], ['t', 'h', 'e'], ['V', 'i', 'r', 'g', 'i', 'n'], ['M', 'a', 'r', 'y']]
    cq_word_encode, cq_char_encode = p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary')
    print(cq_char_encode)
