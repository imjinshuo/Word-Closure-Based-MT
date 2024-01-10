from zh import exp, threshold_f
import numpy as np
import os

names = ['SIT-zh2en-google-closure',
         'CAT-zh2en-google-closure',
         'Purity-zh2en-google-closure',
         'CIT-zh2en-google-closure',
         'PatInv-zh2en-google-closure',
         'SIT-zh2en-bing-closure',
         'CAT-zh2en-bing-closure',
         'Purity-zh2en-bing-closure',
         'CIT-zh2en-bing-closure',
         'PatInv-zh2en-bing-closure',
         'SIT-zh2en-youdao-closure',
         'CAT-zh2en-youdao-closure',
         'Purity-zh2en-youdao-closure',
         'CIT-zh2en-youdao-closure',
         'PatInv-zh2en-youdao-closure',
         ]
thre_dic = {'SIT':0.65, 'CAT':0.65, 'Purity':0.49, 'CIT':0.66, 'PatInv':0.60}
base_google_f1_dic = {'SIT':0.471, 'CAT':0.526, 'Purity':0.331, 'CIT':0.413, 'PatInv':0.000}
base_bing_f1_dic = {'SIT':0.427, 'CAT':0.533, 'Purity':0.419, 'CIT':0.424, 'PatInv':0.000}
base_youdao_f1_dic = {'SIT':0.413, 'CAT':0.529, 'Purity':0.541, 'CIT':0.444, 'PatInv':0.000}
print('ZH2EN:')
os.makedirs('info', exist_ok=True)
os.makedirs('RQ1', exist_ok=True)
f_log = open('RQ1/result_zh2en.txt', 'w')
print('\t'.join(['SUT', 'IT', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', '△F1']))
print('\t'.join(['SUT', 'IT', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', '△F1']), file=f_log)
for name in names:
    config = 13
    IT = name.split('-')[0]
    SUT = name.split('-')[2]
    threshold = thre_dic[name.split('-')[0]]
    base_f1 = 0
    if SUT == 'google':
        base_f1 = base_google_f1_dic[IT]
    elif SUT == 'bing':
        base_f1 = base_bing_f1_dic[IT]
    elif SUT == 'youdao':
        base_f1 = base_youdao_f1_dic[IT]
    file = '../data/RQ1/'+name.split('-')[0]+'-'+name.split('-')[1]+'-'+name.split('-')[2]+'.csv'
    save_file = 'info/'+name+'-'+str(config)+'.npy'
    if name.split('-')[-1] == 'closure':
        TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='WordClosure', IT=IT, opt=True, sem=True, disable_print=True)
    elif name.split('-')[-1] == 'word':
        TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Word', IT=IT, opt=True, sem=True, disable_print=True)
    elif name.split('-')[-1] == 'phrase':
        TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Phrase', IT=IT, opt=True, sem=True, disable_print=True)
    elif name.split('-')[-1] == 'clause':
        TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='Clause', IT=IT, opt=True, sem=True, disable_print=True)
    all_test_info = np.load(save_file, allow_pickle=True)
    all_test_info = all_test_info.tolist()
    TP, FP, TN, FN, precision, recall, f1, accuracy = threshold_f(threshold, all_test_info, IT=IT, output_path='RQ1/' + name +'-'+str(config)+ '.txt')
    print('\t'.join([SUT, IT, str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3)), str(round(f1-base_f1, 3))]))
    print('\t'.join([SUT, IT, str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3)), str(round(f1-base_f1, 3))]), file=f_log)