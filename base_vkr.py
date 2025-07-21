from conllu import parse
import pandas as pd
import os
import collections
import numpy as np
import torch


def get_files_conllu(target_directory):
    return [os.path.join(target_directory, item) for item in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, item)) and item.endswith('conllu')]


def read(nfiles):

    nsentences = []

    for nfile in nfiles:
        # Read the content of your CoNLL-U file
        with open(nfile, "r", encoding="utf-8") as f:
            data = f.read()

        # Parse the CoNLL-U data
        nsentences.append(parse(data))

    sentences = [sent for nsent in nsentences for sent in nsent]

    # Convert to a list of dictionaries for DataFrame creation
    all_tokens = []
    for sentence in sentences:
        for token in sentence:
            all_tokens.append(token)

    # Create a Pandas DataFrame
    df = pd.DataFrame(all_tokens)
    
    return df, sentences, all_tokens


def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    # убрать слишком редкие и слишком частые слова
    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    # отсортировать слова по убыванию частоты
    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    # добавим несуществующее слово с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # если у нас по прежнему слишком много слов, оставить только max_size самых частотных
    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # нумеруем слова
    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    # нормируем частоты слов
    word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return word2id, word2freq


def pos_corpus_to_tensor(df_data, char2id, label2id, max_sent_len, max_token_len):
    sent_count = df_data['sent'].value_counts().shape[0]
    inputs = torch.zeros((sent_count, max_sent_len, max_token_len + 5), dtype=torch.long)
    targets = torch.zeros((sent_count, max_sent_len), dtype=torch.long)

    for row in df_data.values:
        targets[row[0], row[1]] = label2id.get(row[3], 0)
        for char_i, char in enumerate(row[2]):
            inputs[row[0], row[1], char_i + 1] = char2id.get(char, 0)

    return inputs, targets