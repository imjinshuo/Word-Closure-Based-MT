from en import exp, threshold_f
import numpy as np
import os


names = ['SIT-en2zh-LLM-closure',
         'CAT-en2zh-LLM-closure',
         'Purity-en2zh-LLM-closure',
         'CIT-en2zh-LLM-closure',
         'PatInv-en2zh-LLM-closure',
         ]
thre_dic = {'SIT':0.75, 'CAT':0.77, 'Purity':0.63, 'CIT':0.77, 'PatInv':0.75}
print('EN2ZH:')
os.makedirs('info', exist_ok=True)
os.makedirs('LLM', exist_ok=True)
f_log = open('LLM/result_en2zh.txt', 'w')
print('\t'.join(['SUT', 'IT', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1']))
print('\t'.join(['SUT', 'IT', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1']), file=f_log)
for name in names:
    config = 13
    IT = name.split('-')[0]
    SUT = name.split('-')[2]
    threshold = thre_dic[name.split('-')[0]]
    file = '../data/LLM/'+name.split('-')[0]+'-'+name.split('-')[1]+'-'+name.split('-')[2]+'.csv'
    save_file = 'info/'+name+'-'+str(config)+'.npy'
    if name.split('-')[-1] == 'closure':
        TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='WordClosure', IT=IT, opt=True, sem=True, disable_print=True)
    all_test_info = np.load(save_file, allow_pickle=True)
    all_test_info = all_test_info.tolist()
    TP, FP, TN, FN, precision, recall, f1, accuracy = threshold_f(threshold, all_test_info, IT=IT, output_path='LLM/' + name +'-'+str(config)+ '.txt')
    print('\t'.join([SUT, IT, str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3))]))
    print('\t'.join([SUT, IT, str(TP), str(FP), str(TN), str(FN), str(round(accuracy, 3)), str(round(precision, 3)), str(round(recall, 3)), str(round(f1, 3))]), file=f_log)