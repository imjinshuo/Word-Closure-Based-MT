def metric(TP, FP, FN):
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
	return str(round(precision, 3)), str(round(recall, 3)), round(f1, 3)

def compare(file, log):
	f = open(file)
	lines = f.readlines()
	base_positive = 0
	our_positive = 0
	label_positive = 0
	base_TP = 0
	our_TP = 0
	for idx in range(0, len(lines), 14):
		base_s = lines[idx+7].strip().split('\t')
		base_f = lines[idx+8].strip().split('\t')
		our_s = lines[idx+9].strip().split('\t')
		our_f = lines[idx+10].strip().split('\t')
		label_s = lines[idx+11].strip().split('\t')
		label_f = lines[idx+12].strip().split('\t')
		base_positive += len(base_s)
		base_positive += len(base_f)
		our_positive += len(our_s)
		our_positive += len(our_f)
		label_positive += len(label_s)
		label_positive += len(label_f)
		for id in base_s:
			if id in label_s:
				base_TP += 1
		for id in base_f:
			if id in label_f:
				base_TP += 1
		for id in our_s:
			if id in label_s:
				our_TP += 1
		for id in our_f:
			if id in label_f:
				our_TP += 1
	base_FP = base_positive - base_TP
	our_FP = our_positive - our_TP
	base_FN = label_positive - base_TP
	our_FN = label_positive - our_TP

	base_pre, base_rec, base_f1 = metric(base_TP, base_FP, base_FN)
	print(file.split('/')[1].split('-')[0], '\t', base_TP, '\t', base_FP, '\t', base_FN, '\t', base_pre, '\t', base_rec, '\t', str(base_f1))
	print(file.split('/')[1].split('-')[0], '\t', base_TP, '\t', base_FP, '\t', base_FN, '\t', base_pre, '\t', base_rec, '\t', str(base_f1), file=log)
	our_pre, our_rec, our_f1 = metric(our_TP, our_FP, our_FN)
	print('Ours', '\t', our_TP, '\t', our_FP, '\t', our_FN, '\t', our_pre, '\t', our_rec, '\t', our_f1, '\t', str(round(our_f1-base_f1, 3)))
	print('Ours', '\t', our_TP, '\t', our_FP, '\t', our_FN, '\t', our_pre, '\t', our_rec, '\t', our_f1, '\t', str(round(our_f1-base_f1, 3)), file=log)

import os
os.makedirs('RQ3', exist_ok=True)
print('EN2ZH')
f_log1 = open('RQ3/result_en2zh.txt', 'w')
print('\t'.join(['Approach', 'TP_fine', 'FP_fine', 'TN_fine', 'Precision_fine', 'Recall_fine', 'F1_fine', '△F1_fine']))
print('\t'.join(['Approach', 'TP_fine', 'FP_fine', 'TN_fine', 'Precision_fine', 'Recall_fine', 'F1_fine', '△F1_fine']), file=f_log1)
compare('../data/RQ3/CAT-en2zh-google-LABEL.txt', f_log1)
compare('../data/RQ3/Purity-en2zh-google-LABEL.txt', f_log1)
compare('../data/RQ3/CIT-en2zh-google-LABEL.txt', f_log1)
print('ZH2EN')
f_log2 = open('RQ3/result_zh2en.txt', 'w')
print('\t'.join(['Approach', 'TP_fine', 'FP_fine', 'TN_fine', 'Precision_fine', 'Recall_fine', 'F1_fine', '△F1_fine']))
print('\t'.join(['Approach', 'TP_fine', 'FP_fine', 'TN_fine', 'Precision_fine', 'Recall_fine', 'F1_fine', '△F1_fine']), file=f_log2)
compare('../data/RQ3/CAT-zh2en-google-LABEL.txt', f_log2)
compare('../data/RQ3/Purity-zh2en-google-LABEL.txt', f_log2)
compare('../data/RQ3/CIT-zh2en-google-LABEL.txt', f_log2)