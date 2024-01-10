from en import exp
import os

names = ['SIT-en2zh-motivation-closure',
         'CAT-en2zh-motivation-closure',
         'CIT-en2zh-motivation-closure',
         'PatInv-en2zh-motivation-closure',
         'Purity-en2zh-motivation-closure',
         ]
thre_dic = {'SIT':0.75, 'CAT':0.77, 'Purity':0.63, 'CIT':0.77, 'PatInv':0.75}
os.makedirs('info', exist_ok=True)
for name in names:
    config = 13
    IT = name.split('-')[0]
    SUT = name.split('-')[2]
    threshold = thre_dic[name.split('-')[0]]
    file = '../data/Motivation-examples/'+name.split('-')[0]+'-'+name.split('-')[1]+'-'+name.split('-')[2]+'.csv'
    save_file = 'info/'+name+'-'+str(config)+'.npy'
    TP, FP, TN, FN, precision, recall, f1, accuracy = exp(file, save_file, threshold, 'jieba', config=config, clo='WordClosure', IT=IT, opt=True, sem=True, disable_print=False)
