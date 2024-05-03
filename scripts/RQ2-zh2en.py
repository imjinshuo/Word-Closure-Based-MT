from zh import exp, threshold_f
import numpy as np
import os

names = [['SIT-zh2en-merge-closure', 'SIT-zh2en-merge-clause', 'SIT-zh2en-merge-phrase', 'SIT-zh2en-merge-word'],
         ['CAT-zh2en-merge-closure', 'CAT-zh2en-merge-clause', 'CAT-zh2en-merge-phrase', 'CAT-zh2en-merge-word'],
         ['Purity-zh2en-merge-closure', 'Purity-zh2en-merge-clause', 'Purity-zh2en-merge-phrase', 'Purity-zh2en-merge-word'],
         ['CIT-zh2en-merge-closure', 'CIT-zh2en-merge-clause', 'CIT-zh2en-merge-phrase', 'CIT-zh2en-merge-word'],
         ['PatInv-zh2en-merge-closure', 'PatInv-zh2en-merge-clause', 'PatInv-zh2en-merge-phrase', 'PatInv-zh2en-merge-word'],
         ]
thre_dic = {'SIT':0.65, 'CAT':0.65, 'Purity':0.49, 'CIT':0.66, 'PatInv':0.60}
print('ZH2EN:')
os.makedirs('info', exist_ok=True)
os.makedirs('RQ2', exist_ok=True)
f_log = open('RQ2/result_zh2en.txt', 'w')
print('\t'.join(['SUT', 'IT', 'Granularity', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', '△F1']))
print('\t'.join(['SUT', 'IT', 'Granularity', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', '△F1']), file=f_log)
for group in names:
    ours_f1 = 0.0
    for idx, name in enumerate(group):
        config = 13
        IT = name.split('-')[0]
        SUT = name.split('-')[2]
        threshold = thre_dic[name.split('-')[0]]
        file = '../data/RQ2&5/' + name.split('-')[0] + '-' + name.split('-')[1] + '-' + name.split('-')[2] + '.csv'
        save_file = 'info/' + name + '-' + str(config) + '.npy'
        if name.split('-')[-1] in ['closure']:
            TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='WordClosure', IT=IT, opt=True, sem=True, disable_print=True)
        elif name.split('-')[-1] == 'word':
            TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Word', IT=IT, opt=True, sem=True, disable_print=True)
        elif name.split('-')[-1] == 'phrase':
            TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Phrase', IT=IT, opt=True, sem=True, disable_print=True)
        elif name.split('-')[-1] == 'clause':
            TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Clause', IT=IT, opt=True, sem=True, disable_print=True)
        list_sim_thre = [round(num*0.01, 2) for num in range(0, 101)]
        best_f1 = 0
        other_info = ''
        output_file = 'RQ2/'+name+'-'+str(config)+'.tsv'
        f_out = open(output_file, 'w')
        print('\t'.join(['threshold', 'sim_th_v', 'sim_th_b', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'accuracy']), file=f_out)
        f_out.close()
        all_test_info = np.load(save_file, allow_pickle=True)
        all_test_info = all_test_info.tolist()
        for this_threshold in list_sim_thre:
            TP, FP, TN, FN, precision, recall, f1, accuracy = threshold_f(this_threshold, all_test_info, IT=IT)
            this_f_out = open(output_file, 'a')
            print('\t'.join([str(this_threshold), str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3))]), file=this_f_out)
            this_f_out.close()
            if f1 > best_f1:
                best_f1 = f1
                other_info = '\t'.join([str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3))])
        if idx == 0:
            ours_f1 = best_f1
            print('\t'.join([SUT, IT, name.split('-')[-1], other_info]))
            print('\t'.join([SUT, IT, name.split('-')[-1], other_info]), file=f_log)
        else:
            print('\t'.join([SUT, IT, name.split('-')[-1], other_info, str(round(round(ours_f1, 3)-round(best_f1, 3), 3))]))
            print('\t'.join([SUT, IT, name.split('-')[-1], other_info, str(round(round(ours_f1, 3)-round(best_f1, 3), 3))]), file=f_log)