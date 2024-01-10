import jieba
import torch
import itertools
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from nltk.parse import CoreNLPParser
import string
punc = string.punctuation
ch_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.•"
punc = punc + ch_punc
zh_phrase_tagset = ['ADJP', 'ADVP', 'CLP', 'CP', 'DNP', 'DP', 'DVP', 'FRAG', 'IP', 'LCP', 'LST', 'NP', 'PP', 'PRN', 'QP', 'UCP', 'VP', 'CC', 'PU']


def if_punc(token):
    is_punc = True
    for letter in token:
        if letter not in punc:
            is_punc = False
            break
    return is_punc


def load_stopword(file):
    stop_words = []
    f_stopwords = open(file, 'r')
    lines = f_stopwords.readlines()
    for line in lines:
        stop_words.append(line.strip())
    return stop_words
en_stopwords = load_stopword('utils/en_stopwords.txt')
zh_stopwords = load_stopword('utils/cn_stopwords.txt')

parser = CoreNLPParser(url='http://localhost:9000')
zh_parser = CoreNLPParser(url='http://localhost:9001')
device = torch.device('cuda:0')
model_name_or_path = './model_without_co'

config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
model = model_class.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config, ).to(device)

def word_align(lan, src, tgt, tokenizer_name, tokenized=False, if_punc=False, if_stop=False):
    if lan == 'en2zh':
        if isinstance(src, list):
            sent_src = src
            sent_src = [token if token != "n't" else 'not' for token in sent_src]
        else:
            sent_src = src.strip()
            sent_src = list(parser.tokenize(sent_src))
            sent_src = [token if token != "n't" else 'not' for token in sent_src]
        if isinstance(tgt, list):
            sent_tgt = tgt
        else:
            sent_tgt = tgt.strip()
            if tokenizer_name == 'jieba':
                sent_tgt = jieba.lcut(sent_tgt)
            sent_tgt = [token for token in sent_tgt if token != ' ']

        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in
                                                                                 sent_tgt]
        wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src = \
        tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=tokenizer.max_len)[
            'input_ids']
        ids_tgt = \
        tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=tokenizer.max_len)[
            'input_ids']

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_src = [bpe2word_map_src]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        bpe2word_map_tgt = [bpe2word_map_tgt]
        align_layer = 8
        threshold = 1e-3
        model.eval()
        with torch.no_grad():
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, device, 0,
                                                      0, output_prob=True, align_layer=align_layer, extraction='softmax',
                                                      softmax_threshold=threshold, test=True)
        returned_wa = []
        for pair, score in word_aligns_list[0].items():
            if if_punc:
                if sent_src[pair[0]] in punc or sent_tgt[pair[1]] in punc:
                    continue
            if if_stop:
                if sent_src[pair[0]] in en_stopwords or sent_tgt[pair[1]] in zh_stopwords:
                    continue
            returned_wa.append([pair, score.item()])
        return returned_wa, sent_src, sent_tgt
    elif lan == 'zh2en':
        if tokenized:
            sent_src = tgt
            sent_src = [token if token != "n't" else 'not' for token in sent_src]
            sent_tgt = src
        else:
            sent_src = tgt.strip()
            sent_src = list(parser.tokenize(sent_src))
            sent_src = [token if token != "n't" else 'not' for token in sent_src]
            sent_tgt = src.strip()
            if tokenizer_name == 'jieba':
                sent_tgt = jieba.lcut(sent_tgt, cut_all=False)
            sent_tgt = [token for token in sent_tgt if token != ' ']
        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in
                                                                                 sent_tgt]
        wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src = \
            tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                        max_length=tokenizer.max_len)[
                'input_ids']
        ids_tgt = \
            tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                        max_length=tokenizer.max_len)[
                'input_ids']

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_src = [bpe2word_map_src]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        bpe2word_map_tgt = [bpe2word_map_tgt]
        align_layer = 8
        threshold = 1e-3
        model.eval()
        with torch.no_grad():
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, device, 0,
                                                      0, output_prob=True, align_layer=align_layer, extraction='softmax',
                                                      softmax_threshold=threshold, test=True)
        returned_wa = []
        for pair, score in word_aligns_list[0].items():
            if if_punc:
                if sent_src[pair[0]] in punc or sent_tgt[pair[1]] in punc:
                    continue
            if if_stop:
                if sent_src[pair[0]] in en_stopwords or sent_tgt[pair[1]] in zh_stopwords:
                    continue
            returned_wa.append([tuple([pair[1], pair[0]]), score.item()])
        return returned_wa, sent_tgt, sent_src
