import nltk
import string
import difflib
from nltk import Tree
from copy import deepcopy
from itertools import chain
from bert2vec import sim_bert
from nltk.corpus import wordnet
from wordalign import word_align
from pycorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import SmoothingFunction
lemmatizer = WordNetLemmatizer()
chencherry = SmoothingFunction()
punc = string.punctuation
alpha = set('abcdefghijklmnopqrstuvwxyz')
ch_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.•·《》"
punc = punc + ch_punc
nlp_zh = StanfordCoreNLP('http://localhost:9001')
nlp_en = StanfordCoreNLP('http://localhost:9000')


def load_stopword(file):
    stop_words = []
    f_stopwords = open(file, 'r')
    lines = f_stopwords.readlines()
    for line in lines:
        stop_words.append(line.strip())
    return stop_words


en_stopwords = load_stopword('utils/en_stopwords.txt')
zh_stopwords = load_stopword('utils/cn_stopwords.txt')


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def is_PUNCT(token, lang):
    if lang == 'zh':
        if type(token) == list:
            if_stop = True
            for this_token in token:
                this_if_stop = is_PUNCT(this_token, 'zh')
                if not this_if_stop:
                    if_stop = False
            return if_stop
        else:
            if token in punc:
                return True
            else:
                return False
    elif lang == 'en':
        if type(token) == list:
            if_stop = True
            for this_token in token:
                this_if_stop = is_PUNCT(this_token, 'en')
                if not this_if_stop:
                    if_stop = False
            return if_stop
        else:
            if token in punc:
                return True
            else:
                return False


def is_STOP(token, lang):
    if lang == 'zh':
        if type(token) == list:
            if_stop = True
            for this_token in token:
                this_if_stop = is_STOP(this_token, 'zh')
                if not this_if_stop:
                    if_stop = False
            return if_stop
        else:
            if token in zh_stopwords:
                return True
            if_stop = True
            for this_token in token:
                if this_token not in zh_stopwords and this_token not in punc:
                    if_stop = False
            return if_stop
    elif lang == 'en':
        if type(token) == list:
            if_stop = True
            for this_token in token:
                this_if_stop = is_STOP(this_token, 'en')
                if not this_if_stop:
                    if_stop = False
            return if_stop
        else:
            if token.lower() in en_stopwords or token in punc or lemmatizer.lemmatize(token).lower() in en_stopwords:
                return True
            else:
                return False


def get_syn(lang, token):
    if lang == 'zh':
        syn = []
        try:
            wordnet.synsets(token, lang='cmn')
            for each in wordnet.synsets(token, lang='cmn'):
                list = each.lemma_names('cmn')
                for w in list:
                    syn.append(w.replace('+', ''))
        except:
            None
        syn.append(token)
        return syn
    elif lang == 'en':
        syn = wordnet.synsets(token)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in syn]))
        syns = []
        for a_token in lemmas:
            this_token = a_token.split('_')
            for s_token in this_token:
                if s_token.lower() not in syns:
                    syns.append(s_token.lower())
        syns.append(token)
        return syns

import csv
def readFiles(file_name):
    pairs_en_zh = []
    with open(file_name, 'r', encoding='utf-8') as file:  # gbk utf-8
        csvFile = csv.reader(file)
        for lines in csvFile:
            if lines[4] == '1':
                pairs_en_zh.append([lines[0], lines[2], lines[1], lines[3], 1])
            else:
                pairs_en_zh.append([lines[0], lines[2], lines[1], lines[3], 0])
    return pairs_en_zh[1:]


def MAP(word_align_result, reverse=False):
    dict = {}
    index1 = 0
    index2 = 1
    if reverse:
        index1 = 1
        index2 = 0
    for item in word_align_result:
        if item[index1] not in dict:
            dict[item[index1]] = [item[index2]]
        else:
            dict[item[index1]].append(item[index2])
    return dict


def match(list1, list2):
    while True:
        mbs = difflib.SequenceMatcher(None, list1, list2).get_matching_blocks()
        if len(mbs) == 1: break
        for i, j, n in mbs[::-1]:
            if n > 0:
                yield list1[i: i + n]
            del list1[i: i + n]
            del list2[j: j + n]


def valid_index(all_start_index, this_index):
    if all_start_index:
        if this_index < all_start_index[0]:
            return 0
        if this_index > all_start_index[-1]:
            return len(all_start_index)
        for index in range(len(all_start_index) - 1):
            if all_start_index[index] < this_index < all_start_index[index + 1]:
                return index + 1
        raise Exception('find diff wrong !')
    else:
        return 0


def findDifference(en_s_tokens, en_f_tokens):
    same_index_source = []
    same_index_follow = []
    source2follow = {}
    follow2source = {}
    match_output_source = list(
        match([token.lower() for token in en_s_tokens], [token.lower() for token in en_f_tokens]))
    match_output = []
    for match_seq in match_output_source:
        if_matched_s = False
        if_matched_f = False
        for this_index in range(0, len(en_s_tokens) - len(match_seq) + 1):
            if [token.lower() for token in en_s_tokens[this_index:this_index + len(match_seq)]] == match_seq:
                if_matched_s = True
                break
        for this_index in range(0, len(en_f_tokens) - len(match_seq) + 1):
            if [token.lower() for token in en_f_tokens[this_index:this_index + len(match_seq)]] == match_seq:
                if_matched_f = True
                break
        if not if_matched_s or not if_matched_f:
            continue
        match_output.append(match_seq)
    all_len = 0
    for match_seq in match_output:
        all_len += len(match_seq)
    match_output.sort(key=len, reverse=True)
    reached_indexes_s = []
    reached_indexes_f = []
    all_s_start_index = []
    all_f_start_index = []
    for match_seq in match_output:
        this_diff_index_source = []
        this_diff_index_follow = []
        if_s_matched = False
        if_f_matched = False
        this_s_start_index = []
        this_f_start_index = []
        for index in range(0, len(en_s_tokens) - len(match_seq) + 1):
            if index in reached_indexes_s:
                continue
            if [token.lower() for token in en_s_tokens[index:index + len(match_seq)]] == match_seq:
                this_s_start_index.append(index)
                if_s_matched = True
        for index in range(0, len(en_f_tokens) - len(match_seq) + 1):
            if index in reached_indexes_f:
                continue
            if [token.lower() for token in en_f_tokens[index:index + len(match_seq)]] == match_seq:
                this_f_start_index.append(index)
                if_f_matched = True
        if if_s_matched and if_f_matched:
            situations = []
            for this_s_index in this_s_start_index:
                for this_f_index in this_f_start_index:
                    situations.append([this_s_index, this_f_index])
            for this_s_index, this_f_index in situations:
                this_s_index_in_match = valid_index(all_s_start_index, this_s_index)
                this_f_index_in_match = valid_index(all_f_start_index, this_f_index)
                if this_s_index_in_match == this_f_index_in_match:
                    all_s_start_index.insert(this_s_index_in_match, this_s_index)
                    all_f_start_index.insert(this_f_index_in_match, this_f_index)
                    a_index_source = [this_index for this_index in range(this_s_index, this_s_index + len(match_seq))]
                    a_index_follow = [this_index for this_index in range(this_f_index, this_f_index + len(match_seq))]
                    same_index_source.extend(a_index_source)
                    this_diff_index_source.extend(a_index_source)
                    reached_indexes_s.extend(a_index_source)
                    same_index_follow.extend(a_index_follow)
                    this_diff_index_follow.extend(a_index_follow)
                    reached_indexes_f.extend(a_index_follow)
                    break
        for index1, index2 in zip(this_diff_index_source, this_diff_index_follow):
            source2follow[index1] = [index2]
            follow2source[index2] = [index1]
    diff_index_source = [index for index in range(len(en_s_tokens)) if index not in same_index_source]
    diff_index_follow = [index for index in range(len(en_f_tokens)) if index not in same_index_follow]
    diff_index_source.sort()
    diff_index_follow.sort()
    return same_index_source, same_index_follow, diff_index_source, diff_index_follow, source2follow, follow2source


def findOutputDifference(en_s_tokens, en_f_tokens):
    same_index_source = []
    same_index_follow = []
    source2follow = {}
    follow2source = {}
    match_output_source = list(
        match([token.lower() for token in en_s_tokens], [token.lower() for token in en_f_tokens]))
    match_output = []
    for match_seq in match_output_source:
        if_matched_s = False
        if_matched_f = False
        for this_index in range(0, len(en_s_tokens) - len(match_seq) + 1):
            if [token.lower() for token in en_s_tokens[this_index:this_index + len(match_seq)]] == match_seq:
                if_matched_s = True
                break
        for this_index in range(0, len(en_f_tokens) - len(match_seq) + 1):
            if [token.lower() for token in en_f_tokens[this_index:this_index + len(match_seq)]] == match_seq:
                if_matched_f = True
                break
        if not if_matched_s or not if_matched_f:
            continue
        match_output.append(match_seq)
    all_len = 0
    for match_seq in match_output:
        all_len += len(match_seq)
    match_output.sort(key=len, reverse=True)
    reached_indexes_s = []
    reached_indexes_f = []
    all_s_start_index = []
    all_f_start_index = []
    for match_seq in match_output:
        this_diff_index_source = []
        this_diff_index_follow = []
        if_s_matched = False
        if_f_matched = False
        this_s_start_index = []
        this_f_start_index = []
        for index in range(0, len(en_s_tokens) - len(match_seq) + 1):
            if index in reached_indexes_s:
                continue
            if [token.lower() for token in en_s_tokens[index:index + len(match_seq)]] == match_seq:
                this_s_start_index.append(index)
                if_s_matched = True
        for index in range(0, len(en_f_tokens) - len(match_seq) + 1):
            if index in reached_indexes_f:
                continue
            if [token.lower() for token in en_f_tokens[index:index + len(match_seq)]] == match_seq:
                this_f_start_index.append(index)
                if_f_matched = True
        if if_s_matched and if_f_matched:
            situations = []
            for this_s_index in this_s_start_index:
                for this_f_index in this_f_start_index:
                    situations.append([this_s_index, this_f_index])
            for this_s_index, this_f_index in situations:
                this_s_index_in_match = valid_index(all_s_start_index, this_s_index)
                this_f_index_in_match = valid_index(all_f_start_index, this_f_index)
                if this_s_index_in_match == this_f_index_in_match:
                    all_s_start_index.insert(this_s_index_in_match, this_s_index)
                    all_f_start_index.insert(this_f_index_in_match, this_f_index)
                    a_index_source = [this_index for this_index in range(this_s_index, this_s_index + len(match_seq))]
                    a_index_follow = [this_index for this_index in range(this_f_index, this_f_index + len(match_seq))]
                    same_index_source.extend(a_index_source)
                    this_diff_index_source.extend(a_index_source)
                    reached_indexes_s.extend(a_index_source)
                    same_index_follow.extend(a_index_follow)
                    this_diff_index_follow.extend(a_index_follow)
                    reached_indexes_f.extend(a_index_follow)
                    break
        for index1, index2 in zip(this_diff_index_source, this_diff_index_follow):
            source2follow[index1] = index2
            follow2source[index2] = index1
    diff_index_source = [index for index in range(len(en_s_tokens)) if index not in same_index_source]
    diff_index_follow = [index for index in range(len(en_f_tokens)) if index not in same_index_follow]
    same_index_source.sort()
    same_index_follow.sort()
    diff_index_source.sort()
    diff_index_follow.sort()
    return same_index_source, same_index_follow, diff_index_source, diff_index_follow, source2follow, follow2source


def getMappedWord(maps, ids, original_aligned_ids):
    aligned_ids = []
    for id in ids:
        if id in maps:
            aligned_ids.extend(maps[id])
    original_aligned_ids.extend(aligned_ids)
    if original_aligned_ids:
        original_aligned_ids = list(set(original_aligned_ids))
        original_aligned_ids.sort()
    return original_aligned_ids


def buildClosure(closure, s_map, s_map_reverse, f_map, f_map_reverse, source2follow, follow2source):
    original_closure = [[], [], [], []]
    while original_closure != closure:
        original_closure = closure[:]
        closure[1] = getMappedWord(source2follow, closure[0], closure[1][:])
        closure[2] = getMappedWord(s_map, closure[0], closure[2][:])
        closure[0] = getMappedWord(follow2source, closure[1], closure[0][:])
        closure[3] = getMappedWord(f_map, closure[1], closure[3][:])
        closure[0] = getMappedWord(s_map_reverse, closure[2], closure[0][:])
        closure[1] = getMappedWord(f_map_reverse, closure[3], closure[1][:])
    return closure


def align2align(s_align):
    s_align_dict = {}
    for item, score in s_align:
        if item[1] not in s_align_dict:
            s_align_dict[item[1]] = [[score, item[0]]]
        else:
            s_align_dict[item[1]].append([score, item[0]])
    for key, val in s_align_dict.items():
        val.sort(reverse=True)
        high_val = [this_v for this_v in val]
        s_align_dict[key] = high_val
    new_s_align = []
    for key, val in s_align_dict.items():
        for this_val in val:
            new_s_align.append([this_val[1], key])
    return new_s_align


def optimize_align(s_align, f_align, same_index_s, diff_index_source, diff_index_follow, out_source2follow, source2follow, follow2source, s_align_source, f_align_source, en_s_tokens, en_f_tokens, zh_s_tokens, zh_f_tokens):
    s_map_sf = MAP(s_align, reverse=False)
    f_map_sf = MAP(f_align, reverse=False)
    s_align_reverse_dict = {}
    for item, score in s_align_source:
        if item[1] not in s_align_reverse_dict:
            s_align_reverse_dict[item[1]] = [[score, item[0]]]
        else:
            if item[0] not in s_align_reverse_dict[item[1]]:
                s_align_reverse_dict[item[1]].append([score, item[0]])
    f_align_reverse_dict = {}
    for item, score in f_align_source:
        if item[1] not in f_align_reverse_dict:
            f_align_reverse_dict[item[1]] = [[score, item[0]]]
        else:
            if item[0] not in f_align_reverse_dict[item[1]]:
                f_align_reverse_dict[item[1]].append([score, item[0]])
    s_zh_keys = [id for key, val in s_map_sf.items() for id in val]
    f_zh_keys = [id for key, val in f_map_sf.items() for id in val]
    s_keys = [key for key, val in s_map_sf.items()]
    f_keys = [key for key, val in f_map_sf.items()]
    s_zh_keys.sort()
    f_zh_keys.sort()
    notalign_id_s = [id for id in range(len(zh_s_tokens)) if id not in s_zh_keys and not is_PUNCT(zh_s_tokens[id], 'zh')]
    notalign_id_f = [id for id in range(len(zh_f_tokens)) if id not in f_zh_keys and not is_PUNCT(zh_f_tokens[id], 'zh')]
    phrase_tags = ['NP', 'VRD', 'CP', 'CLP', 'ADJP', 'ADVP', 'DNP', 'DP', 'DVP', 'QP', 'VRD']
    if notalign_id_s or notalign_id_f:
        corenlp_s = nlp_zh.annotate(' '.join(zh_s_tokens), properties={
            'annotators': 'pos,depparse,parse',
            'tokenize.whitespace': 'true',
            'ssplit.eolonly': 'true',
            'outputFormat': 'json'
        })
        corenlp_f = nlp_zh.annotate(' '.join(zh_f_tokens), properties={
            'annotators': 'pos,depparse,parse',
            'tokenize.whitespace': 'true',
            'ssplit.eolonly': 'true',
            'outputFormat': 'json'
        })
        s_atree = Tree.fromstring(corenlp_s['sentences'][0]['parse'])
        f_atree = Tree.fromstring(corenlp_f['sentences'][0]['parse'])
        s_atree2 = deepcopy(s_atree)
        f_atree2 = deepcopy(f_atree)
        s_all_leaf = s_atree2.leaves()
        f_all_leaf = f_atree2.leaves()
        for this_index, this_leaf in enumerate(s_all_leaf):
            this_position = s_atree2.leaf_treeposition(this_index)
            s_atree2[this_position[:-1]][0] = str(this_index)
        for this_index, this_leaf in enumerate(f_all_leaf):
            this_position = f_atree2.leaf_treeposition(this_index)
            f_atree2[this_position[:-1]][0] = str(this_index)
        diff_2out_id_s = []
        for key, val in s_map_sf.items():
            if key in diff_index_source:
                diff_2out_id_s.extend(val)
        diff_2out_id_f = []
        for key, val in f_map_sf.items():
            if key in diff_index_follow:
                diff_2out_id_f.extend(val)
        for id in notalign_id_s:
            this_position = s_atree2.leaf_treeposition(id)
            for this_path in range(1, len(this_position)):
                this_sub_tree = s_atree[this_position[:-this_path]]
                this_sub_tree2 = s_atree2[this_position[:-this_path]]
                if len(this_sub_tree.leaves()) != 1:
                    break
            this_leaves = [int(leaf) for leaf in this_sub_tree2.leaves()]
            this_aligned_leaves = [id for id in this_leaves if id in s_zh_keys]
            neig_ids = get_neighbour(id, this_aligned_leaves, zh_s_tokens)
            if not neig_ids and len(this_aligned_leaves) == 1:
                if zh_s_tokens[id] in zh_f_tokens or is_STOP(zh_s_tokens[id], 'zh'):
                    continue
                target_neig = this_aligned_leaves[0]
                add_source_id = []
                for key, val in s_map_sf.items():
                    if target_neig in val:
                        add_source_id.append(key)
                for key in add_source_id:
                    if id not in s_map_sf[key]:
                        s_map_sf[key].append(id)
                        s_zh_keys.append(id)
            elif len(neig_ids) == 1:
                target_neig = neig_ids[0]
                add_source_id = []
                if_diff = False
                if target_neig in diff_2out_id_s:
                    if_diff = True
                for key, val in s_map_sf.items():
                    if target_neig in val:
                        add_source_id.append(key)
                if if_diff:
                    if zh_s_tokens[id] in zh_f_tokens or is_STOP(zh_s_tokens[id], 'zh'):
                        continue
                    for key in add_source_id:
                        if id not in s_map_sf[key]:
                            s_map_sf[key].append(id)
                            s_zh_keys.append(id)
                elif this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_s_tokens[id] not in zh_f_tokens):
                    if is_STOP(zh_s_tokens[id], 'zh') and len(this_leaves) != 2:
                        continue
                    if this_sub_tree._label == 'DNP':
                        continue
                    for key in add_source_id:
                        if id not in s_map_sf[key]:
                            s_map_sf[key].append(id)
                            s_zh_keys.append(id)
            elif len(neig_ids) == 2:
                if neig_ids[0] in diff_2out_id_s:
                    if zh_s_tokens[id] in zh_f_tokens or is_STOP(zh_s_tokens[id], 'zh'):
                        continue
                    target_neig = neig_ids[0]
                    add_source_id = []
                    for key, val in s_map_sf.items():
                        if target_neig in val:
                            add_source_id.append(key)
                    for key in add_source_id:
                        if id not in s_map_sf[key]:
                            s_map_sf[key].append(id)
                            s_zh_keys.append(id)
                elif neig_ids[1] in diff_2out_id_s:
                    if zh_s_tokens[id] in zh_f_tokens or is_STOP(zh_s_tokens[id], 'zh'):
                        continue
                    target_neig = neig_ids[1]
                    add_source_id = []
                    for key, val in s_map_sf.items():
                        if target_neig in val:
                            add_source_id.append(key)
                    for key in add_source_id:
                        if id not in s_map_sf[key]:
                            s_map_sf[key].append(id)
                            s_zh_keys.append(id)
                elif zh_s_tokens[neig_ids[0]] not in ''.join(zh_f_tokens) and zh_s_tokens[neig_ids[1]] in ''.join(zh_f_tokens):
                    if this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_s_tokens[id] not in zh_f_tokens):
                        if is_STOP(zh_s_tokens[id], 'zh') and len(this_leaves) != 2:
                            continue
                        if this_sub_tree._label == 'DNP':
                            continue
                        target_neig = neig_ids[0]
                        add_source_id = []
                        for key, val in s_map_sf.items():
                            if target_neig in val:
                                add_source_id.append(key)
                        for key in add_source_id:
                            if id not in s_map_sf[key]:
                                s_map_sf[key].append(id)
                                s_zh_keys.append(id)
                elif zh_s_tokens[neig_ids[0]] in ''.join(zh_f_tokens) and zh_s_tokens[neig_ids[1]] not in ''.join(zh_f_tokens):
                    if this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_s_tokens[id] not in zh_f_tokens):
                        if is_STOP(zh_s_tokens[id], 'zh') and len(this_leaves) != 2:
                            continue
                        if this_sub_tree._label == 'DNP':
                            continue
                        target_neig = neig_ids[1]
                        add_source_id = []
                        for key, val in s_map_sf.items():
                            if target_neig in val:
                                add_source_id.append(key)
                        for key in add_source_id:
                            if id not in s_map_sf[key]:
                                s_map_sf[key].append(id)
                                s_zh_keys.append(id)
        for id in notalign_id_f:
            this_position = f_atree2.leaf_treeposition(id)
            for this_path in range(1, len(this_position)):
                this_sub_tree = f_atree[this_position[:-this_path]]
                this_sub_tree2 = f_atree2[this_position[:-this_path]]
                if len(this_sub_tree.leaves()) != 1:
                    break
            this_leaves = [int(leaf) for leaf in this_sub_tree2.leaves()]
            this_aligned_leaves = [id for id in this_leaves if id in f_zh_keys]
            neig_ids = get_neighbour(id, this_aligned_leaves, zh_f_tokens)
            if not neig_ids and len(this_aligned_leaves) == 1:
                if zh_f_tokens[id] in zh_s_tokens or is_STOP(zh_f_tokens[id], 'zh'):
                    continue
                target_neig = this_aligned_leaves[0]
                add_source_id = []
                for key, val in f_map_sf.items():
                    if target_neig in val:
                        add_source_id.append(key)
                for key in add_source_id:
                    if id not in f_map_sf[key]:
                        f_map_sf[key].append(id)
                        f_zh_keys.append(id)
            elif len(neig_ids) == 1:
                target_neig = neig_ids[0]
                add_source_id = []
                if_diff = False
                if target_neig in diff_2out_id_f:
                    if_diff = True
                for key, val in f_map_sf.items():
                    if target_neig in val:
                        add_source_id.append(key)
                if if_diff:
                    if zh_f_tokens[id] in zh_s_tokens or is_STOP(zh_f_tokens[id], 'zh'):
                        continue
                    for key in add_source_id:
                        if id not in f_map_sf[key]:
                            f_map_sf[key].append(id)
                            f_zh_keys.append(id)
                elif this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_f_tokens[id] not in zh_s_tokens):
                    if is_STOP(zh_f_tokens[id], 'zh') and len(this_leaves) != 2:
                        continue
                    if this_sub_tree._label == 'DNP':
                        continue
                    for key in add_source_id:
                        if id not in f_map_sf[key]:
                            f_map_sf[key].append(id)
                            f_zh_keys.append(id)
            elif len(neig_ids) == 2:
                if neig_ids[0] in diff_2out_id_f:
                    if zh_f_tokens[id] in zh_s_tokens or is_STOP(zh_f_tokens[id], 'zh'):
                        continue
                    target_neig = neig_ids[0]
                    add_source_id = []
                    for key, val in f_map_sf.items():
                        if target_neig in val:
                            add_source_id.append(key)
                    for key in add_source_id:
                        if id not in f_map_sf[key]:
                            f_map_sf[key].append(id)
                            f_zh_keys.append(id)
                elif neig_ids[1] in diff_2out_id_f:
                    if zh_f_tokens[id] in zh_s_tokens or is_STOP(zh_f_tokens[id], 'zh'):
                        continue
                    target_neig = neig_ids[1]
                    add_source_id = []
                    for key, val in f_map_sf.items():
                        if target_neig in val:
                            add_source_id.append(key)
                    for key in add_source_id:
                        if id not in f_map_sf[key]:
                            f_map_sf[key].append(id)
                            f_zh_keys.append(id)
                elif zh_f_tokens[neig_ids[0]] not in ''.join(zh_s_tokens) and zh_f_tokens[neig_ids[1]] in ''.join(zh_s_tokens):
                    if this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_f_tokens[id] not in zh_s_tokens):
                        if is_STOP(zh_f_tokens[id], 'zh') and len(this_leaves) != 2:
                            continue
                        if this_sub_tree._label == 'DNP':
                            continue
                        target_neig = neig_ids[0]
                        add_source_id = []
                        for key, val in f_map_sf.items():
                            if target_neig in val:
                                add_source_id.append(key)
                        for key in add_source_id:
                            if id not in f_map_sf[key]:
                                f_map_sf[key].append(id)
                                f_zh_keys.append(id)
                elif zh_f_tokens[neig_ids[0]] in ''.join(zh_s_tokens) and zh_f_tokens[neig_ids[1]] not in ''.join(zh_s_tokens):
                    if this_sub_tree._label in phrase_tags or (len(this_aligned_leaves) == 1 and zh_f_tokens[id] not in zh_s_tokens):
                        if is_STOP(zh_f_tokens[id], 'zh') and len(this_leaves) != 2:
                            continue
                        if this_sub_tree._label == 'DNP':
                            continue
                        target_neig = neig_ids[1]
                        add_source_id = []
                        for key, val in f_map_sf.items():
                            if target_neig in val:
                                add_source_id.append(key)
                        for key in add_source_id:
                            if id not in f_map_sf[key]:
                                f_map_sf[key].append(id)
                                f_zh_keys.append(id)
    mid_s_align = []
    mid_f_align = []
    for key, val in s_map_sf.items():
        for this_val in val:
            mid_s_align.append([key, this_val])
    for key, val in f_map_sf.items():
        for this_val in val:
            mid_f_align.append([key, this_val])
    s_map = MAP(mid_s_align, reverse=True)
    f_map = MAP(mid_f_align, reverse=True)
    for index_s in range(len(zh_s_tokens)):
        if index_s in same_index_s:
            continue
        this_token = zh_s_tokens[index_s]
        s_id = [idx for idx, token in enumerate(zh_s_tokens) if token == this_token]
        f_id = [idx for idx, token in enumerate(zh_f_tokens) if token == this_token]
        count_s = len(s_id)
        count_f = len(f_id)
        if count_s == 1 and count_f == 1:
            same_index_s.append(index_s)
            out_source2follow[s_id[0]] = f_id[0]
    same_index_s.sort()
    for same_idx in same_index_s:
        same_token = zh_s_tokens[same_idx]
        count_s = len([token for token in zh_s_tokens if token == same_token])
        count_f = len([token for token in zh_f_tokens if token == same_token])
        if count_s != 1 or count_f != 1:
            continue
        if same_idx in s_map and out_source2follow[same_idx] in f_map:
            s_aligned_idx = s_map[same_idx]
            f_aligned_idx = []
            f_aligned_idx2s = []
            for idx in f_map[out_source2follow[same_idx]]:
                f_aligned_idx.append(idx)
                if idx in follow2source:
                    f_aligned_idx2s.append(follow2source[idx][0])
            s_aligned_idx2f = []
            s_aligned_idx_source = []
            for idx in s_map[same_idx]:
                s_aligned_idx_source.append(idx)
                if idx in source2follow:
                    s_aligned_idx2f.append(source2follow[idx][0])
            f_inter = set(s_aligned_idx2f).union(set(f_aligned_idx))
            s_inter = set(f_aligned_idx2s).union(set(s_aligned_idx))
            s_map[same_idx] = list(s_inter)
            f_map[out_source2follow[same_idx]] = list(f_inter)
        elif same_idx in s_map and out_source2follow[same_idx] not in f_map:
            s_aligned_idx = s_map[same_idx]
            if_invalid = False
            s_aligned_idx2f = []
            s_aligned_idx_source = []
            for idx in s_aligned_idx:
                if idx not in source2follow:
                    if_invalid = True
                else:
                    s_aligned_idx2f.append(source2follow[idx][0])
                    s_aligned_idx_source.append(idx)
            f_map[out_source2follow[same_idx]] = s_aligned_idx2f
        elif same_idx not in s_map and out_source2follow[same_idx] in f_map:
            if_invalid = False
            f_aligned_idx = []
            f_aligned_idx_source = []
            for idx in f_map[out_source2follow[same_idx]]:
                if idx not in follow2source:
                    if_invalid = True
                else:
                    f_aligned_idx.append(follow2source[idx][0])
                    f_aligned_idx_source.append(idx)
            s_map[same_idx] = f_aligned_idx
    new_s_align = []
    new_f_align = []
    for key, val in s_map.items():
        for this_val in val:
            new_s_align.append([this_val, key])
    for key, val in f_map.items():
        for this_val in val:
            new_f_align.append([this_val, key])
    return new_s_align, new_f_align


def get_neighbour(id, this_aligned_leaves, s_tokens):
    neig_ids = []
    if id - 1 in this_aligned_leaves and not is_STOP(s_tokens[id-1], 'zh'):
        neig_ids.append(id - 1)
    elif id - 2 in this_aligned_leaves and not is_STOP(s_tokens[id-2], 'zh'):
        neig_ids.append(id - 2)
    elif id - 3 in this_aligned_leaves and not is_STOP(s_tokens[id-3], 'zh'):
        neig_ids.append(id - 3)
    if id + 1 in this_aligned_leaves and not is_STOP(s_tokens[id+1], 'zh'):
        neig_ids.append(id + 1)
    elif id + 2 in this_aligned_leaves and not is_STOP(s_tokens[id+2], 'zh'):
        neig_ids.append(id + 2)
    elif id + 3 in this_aligned_leaves and not is_STOP(s_tokens[id+3], 'zh'):
        neig_ids.append(id + 3)
    return neig_ids


def traverse(tree, phrases, phrase):
    for idx, subtree in enumerate(tree):
        this_phrase = phrase[:]
        this_phrase.append(idx)
        if type(subtree) == nltk.tree.Tree and subtree._label != 'POS':
            if len(subtree) != 1:
                phrases.append(this_phrase)
            traverse(subtree, phrases, this_phrase)


def get_phrase(tokens):
    corenlp = nlp_zh.annotate(' '.join(tokens), properties={
        'annotators': 'pos,depparse,parse',
        'tokenize.whitespace': 'true',
        'ssplit.eolonly': 'true',
        'outputFormat': 'json'
    })
    tree = Tree.fromstring(corenlp['sentences'][0]['parse'])
    tree2 = deepcopy(tree)
    all_leaf = tree2.leaves()
    for this_index, this_leaf in enumerate(all_leaf):
        this_position = tree2.leaf_treeposition(this_index)
        tree2[this_position[:-1]][0] = str(this_index)
        tree2[this_position[:-1]]._label = 'POS'
    phrases = []
    phrase = []
    traverse(tree2, phrases, phrase)
    phrases.sort()
    phrase_ids = []
    covered_ids = []
    for phrase in phrases:
        leaves = [int(leaf) for leaf in tree2[phrase].leaves()]
        phrase_ids.append(leaves)
        covered_ids.extend(leaves)
    all_units_ids = phrase_ids[:]
    all_units_ids.sort()
    return all_units_ids


def clause_traverse(tree, phrases, phrase):
    for idx, subtree in enumerate(tree):
        this_phrase = phrase[:]
        this_phrase.append(idx)
        if type(subtree) == nltk.tree.Tree:
            if subtree._label in ['IP']:
                phrases.append(this_phrase)
            clause_traverse(subtree, phrases, this_phrase)


def get_clause(tokens):
    corenlp = nlp_zh.annotate(' '.join(tokens), properties={
        'annotators': 'pos,depparse,parse',
        'tokenize.whitespace': 'true',
        'ssplit.eolonly': 'true',
        'outputFormat': 'json'
    })
    tree = Tree.fromstring(corenlp['sentences'][0]['parse'])
    tree2 = deepcopy(tree)
    all_leaf = tree2.leaves()
    for this_index, this_leaf in enumerate(all_leaf):
        this_position = tree2.leaf_treeposition(this_index)
        tree2[this_position[:-1]][0] = str(this_index)
        tree2[this_position[:-1]]._label = 'POS'
    phrases = []
    phrase = []
    clause_traverse(tree2, phrases, phrase)
    phrases.sort()
    phrase_ids = []
    covered_ids = []
    for phrase in phrases:
        leaves = [int(leaf) for leaf in tree2[phrase].leaves()]
        phrase_ids.append(leaves)
        covered_ids.extend(leaves)
    all_units_ids = phrase_ids[:]
    all_units_ids.sort()
    return all_units_ids


def align(s_line, f_line, trans_s_line, trans_f_line, tokenizer, clo=True, opt=True):
    # s_phrases, s_tokens = key_ids(s_line)
    # f_phrases, f_tokens = key_ids(f_line)
    s_align_source, en_s_tokens, zh_s_tokens = word_align('en2zh', s_line, trans_s_line, tokenizer, tokenized=False,
                                                   if_punc=True, if_stop=False)
    f_align_source, en_f_tokens, zh_f_tokens = word_align('en2zh', f_line, trans_f_line, tokenizer, tokenized=False,
                                                   if_punc=True, if_stop=False)

    s_align = align2align(s_align_source)
    f_align = align2align(f_align_source)
    same_index_s, same_index_f, diff_index_s, diff_index_f, out_source2follow, out_follow2source = findOutputDifference(zh_s_tokens, zh_f_tokens)
    same_index_source, same_index_follow, diff_index_source, diff_index_follow, source2follow, follow2source = findDifference(en_s_tokens[:], en_f_tokens[:])

    if opt:
        s_align, f_align = optimize_align(s_align, f_align, same_index_s,
                                          diff_index_source, diff_index_follow, out_source2follow,
                                          source2follow, follow2source, s_align_source, f_align_source,
                                          en_s_tokens, en_f_tokens, zh_s_tokens, zh_f_tokens)

    s_map = MAP(s_align, reverse=False)
    s_map_reverse = MAP(s_align, reverse=True)
    f_map = MAP(f_align, reverse=False)
    f_map_reverse = MAP(f_align, reverse=True)

    if clo == 'WordClosure':
        s_indexes = [index for index in range(len(en_s_tokens))]
        f_indexes = [index for index in range(len(en_f_tokens))]
        all_closures = []
        for index in s_indexes:
            this_closure = buildClosure([[index], [], [], []], s_map, s_map_reverse, f_map, f_map_reverse, source2follow, follow2source)
            if this_closure not in all_closures:
                all_closures.append(this_closure)
        for index in f_indexes:
            this_closure = buildClosure([[], [index], [], []], s_map, s_map_reverse, f_map, f_map_reverse, source2follow, follow2source)
            if this_closure not in all_closures:
                all_closures.append(this_closure)
        constructed_closure = []
        for closure in all_closures:
            if closure[0] != [] and closure[2] == []:
                continue
            if closure[1] != [] and closure[3] == []:
                continue
            constructed_closure.append(closure)
        closures_for_comparison = []
        closures_not_for_comparison = []
        for closure in all_closures:
            inter_s = set(closure[0]).intersection(set(diff_index_source))
            inter_f = set(closure[1]).intersection(set(diff_index_follow))
            if inter_s or inter_f:
                closures_not_for_comparison.append(closure)
            elif closure in constructed_closure:
                closures_for_comparison.append(closure)
    elif clo == 'Word':
        closures_for_comparison = []
        closures_not_for_comparison = []
        closures_not_idx_s = []
        closures_not_idx_f = []
        for idx_s, unit_s in enumerate(zh_s_tokens):
            if idx_s in s_map_reverse:
                intersect = list(set(s_map_reverse[idx_s]).intersection(set(diff_index_source)))
                if intersect:
                    closures_not_for_comparison.append([s_map_reverse[idx_s], [], [idx_s], []])
                    closures_not_idx_s.append(idx_s)
        for idx_f, unit_f in enumerate(zh_f_tokens):
            if idx_f in f_map_reverse:
                intersect = list(set(f_map_reverse[idx_f]).intersection(set(diff_index_follow)))
                if intersect:
                    closures_not_for_comparison.append([[], f_map_reverse[idx_f], [], [idx_f]])
                    closures_not_idx_f.append(idx_f)
        for idx_s, unit_s in enumerate(zh_s_tokens):
            if idx_s in s_map_reverse and idx_s not in closures_not_idx_s:
                for idx_f, unit_f in enumerate(zh_f_tokens):
                    if idx_f in f_map_reverse and idx_f not in closures_not_idx_f:
                        aligned_ids_s = s_map_reverse[idx_s]
                        aligned_ids_s2f = [source2follow[this_s_aligned][0] for this_s_aligned in aligned_ids_s if this_s_aligned in source2follow]
                        aligned_ids_f = f_map_reverse[idx_f]
                        if aligned_ids_s2f == aligned_ids_f:
                            closures_for_comparison.append([aligned_ids_s, aligned_ids_f, [idx_s], [idx_f]])
    elif clo == 'Phrase':
        phrases_s = get_phrase(zh_s_tokens)
        phrases_f = get_phrase(zh_f_tokens)
        closures_for_comparison = []
        closures_not_for_comparison = []
        closures_not_idx_s = []
        closures_not_idx_f = []
        for idx_s, unit_s in enumerate(phrases_s):
            aligned_ids = []
            for idx in unit_s:
                if idx in s_map_reverse:
                    aligned_ids.extend(s_map_reverse[idx])
            aligned_ids = list(set(aligned_ids))
            aligned_ids.sort()
            intersect = list(set(aligned_ids).intersection(set(diff_index_source)))
            if intersect:
                closures_not_for_comparison.append([aligned_ids, [], unit_s, []])
                closures_not_idx_s.append(idx_s)
        for idx_f, unit_f in enumerate(phrases_f):
            aligned_ids = []
            for idx in unit_f:
                if idx in f_map_reverse:
                    aligned_ids.extend(f_map_reverse[idx])
            aligned_ids = list(set(aligned_ids))
            aligned_ids.sort()
            intersect = list(set(aligned_ids).intersection(set(diff_index_follow)))
            if intersect:
                closures_not_for_comparison.append([[], aligned_ids, [], unit_f])
                closures_not_idx_f.append(idx_f)
        for idx_s, unit_s in enumerate(phrases_s):
            aligned_ids_s = []
            for idx in unit_s:
                if idx in s_map_reverse:
                    aligned_ids_s.extend(s_map_reverse[idx])
            aligned_ids_s = list(set(aligned_ids_s))
            aligned_ids_s.sort()
            if aligned_ids_s and idx_s not in closures_not_idx_s:
                for idx_f, unit_f in enumerate(phrases_f):
                    aligned_ids_f = []
                    for idx in unit_f:
                        if idx in f_map_reverse:
                            aligned_ids_f.extend(f_map_reverse[idx])
                    aligned_ids_f = list(set(aligned_ids_f))
                    aligned_ids_f.sort()
                    if aligned_ids_f and idx_f not in closures_not_idx_f:
                        aligned_ids_s2f = [source2follow[this_s_aligned][0] for this_s_aligned in aligned_ids_s if this_s_aligned in source2follow]
                        if aligned_ids_s2f == aligned_ids_f:
                            closures_for_comparison.append([aligned_ids_s, aligned_ids_f, unit_s, unit_f])
    elif clo == 'Clause':
        phrases_s = get_clause(zh_s_tokens)
        phrases_f = get_clause(zh_f_tokens)
        closures_for_comparison = []
        closures_not_for_comparison = []
        closures_not_idx_s = []
        closures_not_idx_f = []
        for idx_s, unit_s in enumerate(phrases_s):
            aligned_ids = []
            for idx in unit_s:
                if idx in s_map_reverse:
                    aligned_ids.extend(s_map_reverse[idx])
            aligned_ids = list(set(aligned_ids))
            aligned_ids.sort()
            intersect = list(set(aligned_ids).intersection(set(diff_index_source)))
            if intersect:
                closures_not_for_comparison.append([aligned_ids, [], unit_s, []])
                closures_not_idx_s.append(idx_s)
        for idx_f, unit_f in enumerate(phrases_f):
            aligned_ids = []
            for idx in unit_f:
                if idx in f_map_reverse:
                    aligned_ids.extend(f_map_reverse[idx])
            aligned_ids = list(set(aligned_ids))
            aligned_ids.sort()
            intersect = list(set(aligned_ids).intersection(set(diff_index_follow)))
            if intersect:
                closures_not_for_comparison.append([[], aligned_ids, [], unit_f])
                closures_not_idx_f.append(idx_f)
        for idx_s, unit_s in enumerate(phrases_s):
            aligned_ids_s = []
            for idx in unit_s:
                if idx in s_map_reverse:
                    aligned_ids_s.extend(s_map_reverse[idx])
            aligned_ids_s = list(set(aligned_ids_s))
            aligned_ids_s.sort()
            if aligned_ids_s and idx_s not in closures_not_idx_s:
                for idx_f, unit_f in enumerate(phrases_f):
                    aligned_ids_f = []
                    for idx in unit_f:
                        if idx in f_map_reverse:
                            aligned_ids_f.extend(f_map_reverse[idx])
                    aligned_ids_f = list(set(aligned_ids_f))
                    aligned_ids_f.sort()
                    if aligned_ids_f and idx_f not in closures_not_idx_f:
                        aligned_ids_s2f = [source2follow[this_s_aligned][0] for this_s_aligned in aligned_ids_s if this_s_aligned in source2follow]
                        if aligned_ids_s2f == aligned_ids_f:
                            closures_for_comparison.append([aligned_ids_s, aligned_ids_f, unit_s, unit_f])
    return closures_for_comparison, closures_not_for_comparison, \
           en_s_tokens, en_f_tokens, \
           zh_s_tokens, zh_f_tokens, \
           s_align, f_align

