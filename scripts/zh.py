import os, sys
from tqdm import tqdm
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tempdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tempdir+'/utils')
from utils.zh_utils import *
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity
w2v = models.KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)


def get_vector(tokens):
    return np.sum(np.array([w2v[i] for i in tokens]), axis=0)


def w2v_sim(token1_list, token2_list):
    if token1_list == token2_list:
        return 1.0
    try:
        this_sim = cosine_similarity(get_vector(token1_list).reshape(1, -1), get_vector(token2_list).reshape(1, -1))
        return this_sim[0][0]
    except:
        return 0.0


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


def syn(token1_list, token2_list):
    lower_token1_list = [token.lower() for token in token1_list]
    lower_token2_list = [token.lower() for token in token2_list]
    if set(lower_token1_list) == set(lower_token2_list):
        return 1.0
    token1_list2 = [lemmatizer.lemmatize(token.lower()) for token in token1_list if token not in en_stopwords]
    if not token1_list2:
        token1_list2 = token1_list[:]
    token2_list2 = [lemmatizer.lemmatize(token.lower()) for token in token2_list if token not in en_stopwords]
    if not token2_list2:
        token2_list2 = token2_list[:]
    if set(token1_list2) == set(token2_list2):
        return 1.0
    s_syn = []
    f_syn = []
    for token1 in token1_list2:
        this_syn = get_syn('en', token1)
        s_syn.extend(this_syn)
    for token2 in token2_list2:
        this_syn = get_syn('en', token2)
        f_syn.extend(this_syn)
    if set(token1_list2).issubset(set(f_syn)) or set(token2_list2).issubset(set(s_syn)):
        return 1.0
    else:
        return 0.0


def exp(file, save_file, threshold, tokenizer, config=13, clo='WordClosure', IT='', opt=True, sem=True, disable_print=False):
    if disable_print:
        sys.stdout = open(os.devnull, 'w')
    else:
        sys.stdout = sys.__stdout__
    all_test_info = []
    trans_pairs = readFiles(file)
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    num = 0
    i_index_list = [i for i in range(len(trans_pairs))]
    for i in tqdm(i_index_list):
        if_same = True
        s_line = trans_pairs[i][0]
        trans_s_line = trans_pairs[i][1]
        f_line = trans_pairs[i][2]
        trans_f_line = trans_pairs[i][3]
        if_violate = trans_pairs[i][4]

        closures_for_comparison, closures_not_for_comparison, \
        en_s_tokens, en_f_tokens, \
        zh_s_tokens, zh_f_tokens, \
        s_align, f_align = align(s_line, f_line, trans_s_line, trans_f_line, tokenizer, clo=clo, opt=opt)

        sim_reached_idx_s = set([])
        sim_reached_idx_f = set([])
        for this_closureid, this_closure in enumerate(closures_for_comparison):
            for idx in this_closure[2]:
                sim_reached_idx_s.add(idx)
            for idx in this_closure[3]:
                sim_reached_idx_f.add(idx)
        sim_reached_uncompared_idx_s = set([])
        sim_reached_uncompared_idx_f = set([])
        for this_closureid, this_closure in enumerate(closures_not_for_comparison):
            for idx in this_closure[2]:
                sim_reached_uncompared_idx_s.add(idx)
            for idx in this_closure[3]:
                sim_reached_uncompared_idx_f.add(idx)
        same_zh_s_tokens = [zh_s_tokens[id] for id in sim_reached_idx_s]
        same_zh_f_tokens = [zh_f_tokens[id] for id in sim_reached_idx_f]
        diff_zh_s_tokens = [lemmatizer.lemmatize(zh_s_tokens[id].lower()) for id in sim_reached_uncompared_idx_s]
        diff_zh_f_tokens = [lemmatizer.lemmatize(zh_f_tokens[id].lower()) for id in sim_reached_uncompared_idx_f]
        not_compare_tokens = diff_zh_s_tokens[:]
        not_compare_tokens.extend(diff_zh_f_tokens)
        not_included_idx_s = [idx for idx in range(len(zh_s_tokens)) if
                              idx not in sim_reached_idx_s and idx not in sim_reached_uncompared_idx_s]
        not_included_idx_f = [idx for idx in range(len(zh_f_tokens)) if
                              idx not in sim_reached_idx_f and idx not in sim_reached_uncompared_idx_f]

        RED_sim_tokenpair = []
        for this_closureid, this_closure in enumerate(closures_for_comparison):
            source_tokens1 = [en_s_tokens[this_index] for this_index in this_closure[0]]
            follow_tokens1 = [en_f_tokens[this_index] for this_index in this_closure[1]]
            source_tokens = [zh_s_tokens[this_index] for this_index in this_closure[2]]
            follow_tokens = [zh_f_tokens[this_index] for this_index in this_closure[3]]
            RED_sim_tokenpair.append([source_tokens1, follow_tokens1, source_tokens, follow_tokens])
        print(f'{color.BOLD}S_s{color.END}: {s_line}')
        print(f'{color.BOLD}S_f{color.END}: {f_line}')
        print(f'{color.BOLD}T_s{color.END}: {zh_s_tokens}')
        print(f'{color.BOLD}T_f{color.END}: {zh_f_tokens}')

        this_diff_score = 0.0
        dissim_threshold = 0.96
        s_diff_words_ids = []
        f_diff_words_ids = []
        if IT == 'PatInv':
            for closure in closures_not_for_comparison:
                s_diff_words_ids.extend(closure[2])
                f_diff_words_ids.extend(closure[3])
            s_diff_words_ids.sort()
            f_diff_words_ids.sort()
            s_diff_words = [zh_s_tokens[this_index] for this_index in s_diff_words_ids]
            f_diff_words = [zh_f_tokens[this_index] for this_index in f_diff_words_ids]
            if s_diff_words_ids and f_diff_words_ids:
                if sem:
                    if config == 1:
                        if syn(s_diff_words, f_diff_words):
                            this_diff_score = 1.0
                    elif config == 2:
                        this_diff_score = w2v_sim(s_diff_words, f_diff_words)
                    elif config == 3:
                        this_diff_score = sim_bert('en', zh_s_tokens, zh_f_tokens, s_diff_words_ids, f_diff_words_ids)
                    elif config == 12:
                        this_diff_score = w2v_sim(s_diff_words, f_diff_words)
                        if syn(s_diff_words, f_diff_words):
                            this_diff_score = 1.0
                    elif config == 13:
                        this_diff_score = sim_bert('en', zh_s_tokens, zh_f_tokens, s_diff_words_ids, f_diff_words_ids)
                        if syn(s_diff_words, f_diff_words):
                            this_diff_score = 1.0
                else:
                    if set(s_diff_words) == set(f_diff_words):
                        this_diff_score = 0.0
                if set(s_diff_words).issubset(set(f_diff_words)) or set(f_diff_words).issubset(set(s_diff_words)):
                    this_diff_score = 0.0
                if this_diff_score > dissim_threshold:
                    if_same = False
            else:
                this_diff_score = 0.0
        closure_score = []
        diff_tokens = []
        for this_closureid, this_closure in enumerate(closures_for_comparison):
            s_pair_tokens = [zh_s_tokens[this_index] for this_index in this_closure[2]]
            f_pair_tokens = [zh_f_tokens[this_index] for this_index in this_closure[3]]
            # if len(s_pair_tokens)!=1 and len(f_pair_tokens)!=1 and s_pair_tokens != f_pair_tokens:
            #   print('')\
            this_score = 0.0
            if sem:
                if config == 1:
                    if syn(s_pair_tokens, f_pair_tokens):
                        this_score = 1.0
                elif config == 2:
                    this_score = w2v_sim(s_pair_tokens, f_pair_tokens)
                elif config == 3:
                    this_score = sim_bert('en', zh_s_tokens, zh_f_tokens, this_closure[2], this_closure[3])
                elif config == 12:
                    this_score = w2v_sim(s_pair_tokens, f_pair_tokens)
                    if syn(s_pair_tokens, f_pair_tokens):
                        this_score = 1.0
                elif config == 13:
                    this_score = sim_bert('en', zh_s_tokens, zh_f_tokens, this_closure[2], this_closure[3])
                    if syn(s_pair_tokens, f_pair_tokens):
                        this_score = 1.0
            else:
                if set(s_pair_tokens) == set(f_pair_tokens):
                    this_score = 1.0
            if is_STOP(s_pair_tokens, 'en') and is_STOP(f_pair_tokens, 'en'):
                this_score = 1.0
            if s_pair_tokens == diff_zh_s_tokens and f_pair_tokens == diff_zh_f_tokens:
                this_score = 1.0
            closure_score.append([s_pair_tokens, f_pair_tokens, this_score, this_closure[2], this_closure[3]])
            if this_score <= threshold:
                diff_tokens.append([s_pair_tokens, f_pair_tokens, this_score])
        if diff_tokens:
            if_same = False

        repeated_tokens_s = []
        for token in zh_s_tokens:
            count_num = len([i for i in zh_s_tokens if i == token])
            if count_num > 1:
                repeated_tokens_s.append(token)
        repeated_tokens_f = []
        for token in zh_f_tokens:
            count_num = len([i for i in zh_f_tokens if i == token])
            if count_num > 1:
                repeated_tokens_f.append(token)
        not_included_zh_s_idx = [id for id in not_included_idx_s if zh_s_tokens[id] not in punc
                                 if not is_STOP(zh_s_tokens[id], 'en')
                                 and zh_s_tokens[id] not in repeated_tokens_s and zh_s_tokens[id].lower() not in trans_f_line.lower()]# and zh_s_tokens[id] not in STOP_align_tokens_s]
        not_included_zh_f_idx = [id for id in not_included_idx_f if zh_f_tokens[id] not in punc
                                 if not is_STOP(zh_f_tokens[id], 'en')
                                 and zh_f_tokens[id] not in repeated_tokens_f and zh_f_tokens[id].lower() not in trans_s_line.lower()]# and zh_f_tokens[id] not in STOP_align_tokens_f]
        not_included_tokens_s = [zh_s_tokens[id] for id in not_included_zh_s_idx]
        not_included_tokens_f = [zh_f_tokens[id] for id in not_included_zh_f_idx]
        score_map = []
        if not_included_tokens_s and not_included_tokens_f:
            for token1, idx1 in zip(not_included_tokens_s, not_included_zh_s_idx):
                for token2, idx2 in zip(not_included_tokens_f, not_included_zh_f_idx):
                    same_score = 0.0
                    if sem:
                        if config == 1:
                            if syn([token1], [token2]):
                                same_score = 1.0
                        elif config == 2:
                            same_score = w2v_sim([token1], [token2])
                        elif config == 3:
                            same_score = sim_bert('en', zh_s_tokens, zh_f_tokens, [idx1], [idx2])
                        elif config == 12:
                            same_score = w2v_sim([token1], [token2])
                            if syn([token1], [token2]):
                                same_score = 1.0
                        elif config == 13:
                            same_score = sim_bert('en', zh_s_tokens, zh_f_tokens, [idx1], [idx2])
                            if syn([token1], [token2]):
                                same_score = 1.0
                    else:
                        if token1 == token2:
                            same_score = 1.0
                    score_map.append([same_score, token1, token2, idx1, idx2])
        score_map.sort(reverse=True)
        delete_tokens_s = []
        delete_tokens_f = []
        for item in score_map:
            score = item[0]
            token1 = item[1]
            token2 = item[2]
            if token1 in delete_tokens_s or token2 in delete_tokens_f:
                continue
            if score > threshold:
                delete_tokens_s.append(token1)
                delete_tokens_f.append(token2)
        not_included_diff_score_s = [item for item in not_included_tokens_s if item not in delete_tokens_s]
        not_included_diff_score_f = [item for item in not_included_tokens_f if item not in delete_tokens_f]

        if not_included_diff_score_s or not_included_diff_score_f:
            if_same = False

        White_output_s = []
        White_output_f = []
        for align_pair in s_align:
            White_output_s.append([align_pair[0], en_s_tokens[align_pair[0]], zh_s_tokens[align_pair[1]]])
        for align_pair in f_align:
            White_output_f.append([align_pair[0], en_f_tokens[align_pair[0]], zh_f_tokens[align_pair[1]]])
        White_output_s.sort()
        White_output_f.sort()
        print(f'{color.BOLD}M_s{color.END}: {color.BOLD}{color.GREEN}{White_output_s}{color.END}')
        print(f'{color.BOLD}M_f{color.END}: {color.BOLD}{color.GREEN}{White_output_f}{color.END}')
        print(f'{color.BOLD}CWC{color.END}: {color.BOLD}{color.BLUE}{closure_score}{color.END}')
        if diff_tokens:
            print(f'{color.BOLD}Violated CWC{color.END}: {color.BOLD}{color.RED}{diff_tokens}{color.END}')
        # print(not_included_tokens_s)
        # print(not_included_tokens_f)
        if not_included_tokens_s:
            print(f"{color.BOLD}UWC{color.END}: {color.BOLD}{color.BLUE}{not_included_tokens_s}{color.END}")
        if not_included_diff_score_s:
            print(f"{color.BOLD}Violated UWC{color.END}: {color.BOLD}{color.RED}{not_included_diff_score_s}{color.END}")
        if not_included_tokens_f:
            print(f"{color.BOLD}UWC{color.END}: {color.BOLD}{color.BLUE}{not_included_tokens_f}{color.END}")
        if not_included_diff_score_f:
            print(f"{color.BOLD}Violated UWC{color.END}: {color.BOLD}{color.RED}{not_included_diff_score_f}{color.END}")
        if IT == 'PatInv':
            print(
                f"{color.BOLD}MWC{color.END}: {color.BOLD}{color.BLUE}{diff_zh_s_tokens}, {diff_zh_f_tokens} (similarity:{this_diff_score}){color.END}")
            if this_diff_score > dissim_threshold:
                print(
                    f"{color.BOLD}Violated MWC{color.END}: {color.BOLD}{color.RED}{diff_zh_s_tokens}, {diff_zh_f_tokens} (similarity:{this_diff_score}){color.END}")

        if not if_same:
            if if_violate:
                TP += 1
                print(f'{color.BOLD}{color.YELLOW}{"TP"}{color.END}')
            else:
                FP += 1
                print(f'{color.BOLD}{color.YELLOW}{"FP"}{color.END}')
            print(f'{color.BOLD}{color.PURPLE}{FP, TP}{color.END}')
            print(f'{color.BOLD}{color.DARKCYAN}{FN, TN}{color.END}')
            num += 1
        else:
            print(i)
            if if_violate:
                FN += 1
                print(f'{color.BOLD}{color.CYAN}{"FN"}{color.END}')
            else:
                TN += 1
                print(f'{color.BOLD}{color.CYAN}{"TN"}{color.END}')
            print(f'{color.BOLD}{color.PURPLE}{FP, TP}{color.END}')
            print(f'{color.BOLD}{color.DARKCYAN}{FN, TN}{color.END}')
            num += 1
            print(i, num)
        # if FN == 13:
        #     break
        test_info = {}
        test_info['violate'] = if_violate
        test_info['en_s'] = trans_pairs[i][0]
        test_info['en_f'] = trans_pairs[i][2]
        test_info['zh_s'] = trans_pairs[i][1]
        test_info['zh_f'] = trans_pairs[i][3]
        test_info['zh_s_tokens'] = zh_s_tokens
        test_info['zh_f_tokens'] = zh_f_tokens
        test_info['closure_score'] = closure_score
        test_info['score_map'] = score_map
        test_info['this_diff_score'] = this_diff_score
        test_info['not_included_tokens_s'] = not_included_tokens_s
        test_info['not_included_tokens_f'] = not_included_tokens_f
        test_info['not_included_zh_s_idx'] = not_included_zh_s_idx
        test_info['not_included_zh_f_idx'] = not_included_zh_f_idx
        test_info['trans_f_line'] = trans_f_line
        test_info['trans_s_line'] = trans_s_line
        test_info['s_diff_words_ids'] = s_diff_words_ids
        test_info['f_diff_words_ids'] = f_diff_words_ids
        all_test_info.append(test_info)
    print('FP: ', FP)
    print('TP: ', TP)
    print('FN: ', FN)
    print('TN: ', TN)
    np.save(save_file, all_test_info)
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP == 0 and FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)
    print('accuracy: ', accuracy)
    sys.stdout = sys.__stdout__
    return TP, FP, TN, FN, precision, recall, f1, accuracy


def threshold_f(threshold, all_test_info, IT='', output_path=None):
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    violation_info = []
    num = -1
    for test_info in all_test_info:
        num += 1
        vio_info_id_s = []
        vio_info_id_f = []
        en_s = test_info['en_s']
        en_f = test_info['en_f']
        zh_s = test_info['zh_s']
        zh_f = test_info['zh_f']
        zh_s_tokens = test_info['zh_s_tokens']
        zh_f_tokens = test_info['zh_f_tokens']
        closure_score = test_info['closure_score']
        score_map = test_info['score_map']
        not_included_tokens_s = test_info['not_included_tokens_s']
        not_included_tokens_f = test_info['not_included_tokens_f']
        not_included_zh_s_idx = test_info['not_included_zh_s_idx']
        not_included_zh_f_idx = test_info['not_included_zh_f_idx']
        s_diff_words_ids = test_info['s_diff_words_ids']
        f_diff_words_ids = test_info['f_diff_words_ids']
        if_vio = test_info['violate']
        this_diff_score = test_info['this_diff_score']
        dissim_threshold = 0.96
        if_same = True
        if IT == 'PatInv':
            if this_diff_score > dissim_threshold:
                if_same = False
                vio_info_id_s.extend(s_diff_words_ids)
                vio_info_id_f.extend(f_diff_words_ids)
        for score in closure_score:
            if score[2] <= threshold:
                if_same = False
                vio_info_id_s.extend(score[3])
                vio_info_id_f.extend(score[4])
        delete_tokens_s = []
        delete_tokens_f = []
        for item in score_map:
            score = item[0]
            token1 = item[1]
            token2 = item[2]
            if token1 in delete_tokens_s or token2 in delete_tokens_f:
                continue
            if score > threshold:
                delete_tokens_s.append(token1)
                delete_tokens_f.append(token2)
        not_included_diff_score_s = [item for item in not_included_tokens_s if item not in delete_tokens_s]
        not_included_diff_score_f = [item for item in not_included_tokens_f if item not in delete_tokens_f]
        not_included_diff_score_s_idx = [idx for item, idx in zip(not_included_tokens_s, not_included_zh_s_idx) if item not in delete_tokens_s]
        not_included_diff_score_f_idx = [idx for item, idx in zip(not_included_tokens_f, not_included_zh_f_idx) if item not in delete_tokens_f]


        if not_included_diff_score_s or not_included_diff_score_f:
            if_same = False
            if not_included_diff_score_s:
                vio_info_id_s.extend(not_included_diff_score_s_idx)
            if not_included_diff_score_f:
                vio_info_id_f.extend(not_included_diff_score_f_idx)

        if if_same:
            if if_vio:
                FN += 1
            else:
                TN += 1
        else:
            if if_vio:
                TP += 1
                vio_info_id_s.sort()
                vio_info_id_f.sort()
                violation_info.append([num, en_s, en_f, zh_s, zh_f, zh_s_tokens, zh_f_tokens, vio_info_id_s, vio_info_id_f])
            else:
                FP += 1
    if output_path:
        f_out = open(output_path, 'w')
        for item in violation_info:
            print(item[0], file=f_out)
            print(item[1], file=f_out)
            print(item[2], file=f_out)
            print(item[3], file=f_out)
            print(item[4], file=f_out)
            print(' '.join(item[5]), file=f_out)
            print(' '.join(item[6]), file=f_out)
            print('\t'.join(str(i) for i in item[7]), file=f_out)
            print('\t'.join(str(i) for i in item[8]), file=f_out)
            print('', file=f_out)
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP == 0 and FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, FP, TN, FN, precision, recall, f1, accuracy

