import os
import csv
CAT_FN = [0, 0]
CAT_FP = [0, 0]
CAT_FP2 = [0, 0]
SIT_FN = [0, 0]
SIT_FN2 = [0, 0]
SIT_FP = [0, 0]
CIT_FN = [0, 0]
CIT_FN2 = [0, 0]
Purity_FN = [0, 0]
RTI_FP2 = [0, 0]
PatInv_FN = [0, 0]
files = os.listdir('../data/Limitation-statistics/')
files.sort()
for file in files:
	lines = []
	with open("../data/Limitation-statistics/" + file, 'r', encoding='utf-8') as f:  # gbk utf-8
		csvFile = csv.reader(f)
		for line in csvFile:
			lines.append([line[0], line[1], line[2], line[3], line[4], line[5]])
	lines = lines[1:]
	num = 0
	sum = 0
	sum2 = 0
	for line in lines:
		num += 1
		if line[4] == 'True':
			sum += 1
		if line[5] == 'True':
			sum2 += 1
	if file.split('-')[0] == 'CAT':
		if file[:-4].split('-')[2] == 'FP':
			CAT_FP[0] += num
			CAT_FP[1] += sum
			CAT_FP2[0] += num
			CAT_FP2[1] += sum2
		elif file[:-4].split('-')[2] == 'FN':
			CAT_FN[0] += num
			CAT_FN[1] += sum
	elif file.split('-')[0] == 'SIT':
		if file[:-4].split('-')[2] == 'FP':
			SIT_FP[0] += num
			SIT_FP[1] += sum
		elif file[:-4].split('-')[2] == 'FN':
			SIT_FN[0] += num
			SIT_FN[1] += sum
			SIT_FN2[0] += num
			SIT_FN2[1] += sum2
	elif file.split('-')[0] == 'CIT':
		if file[:-4].split('-')[2] == 'FN':
			CIT_FN[0] += num
			CIT_FN[1] += sum
			CIT_FN2[0] += num
			CIT_FN2[1] += sum2
	elif file.split('-')[0] == 'Purity':
		if file[:-4].split('-')[2] == 'FN':
			Purity_FN[0] += num
			Purity_FN[1] += sum
		elif file[:-4].split('-')[2] == 'FP':
			RTI_FP2[0] += num
			RTI_FP2[1] += sum2
	elif file.split('-')[0] == 'PatInv':
		if file[:-4].split('-')[2] == 'FN':
			PatInv_FN[0] += num
			PatInv_FN[1] += sum

import os
os.makedirs('Limitation', exist_ok=True)
f_log = open('Limitation/result.txt', 'w')
print(f"{round(round(SIT_FP[1]/SIT_FP[0], 3)*100, 1)}% of FPs in SIT is due to Limitation of Class-A methods.")
print(f"{round(round(CAT_FP[1]/CAT_FP[0], 3)*100, 1)}% of FPs in CAT is due to Limitation of Class-A methods.")
print(f"{round(round(SIT_FN[1]/SIT_FN[0], 3)*100, 1)}% of FNs in SIT is due to Limitation of Class-A methods.")
print(f"{round(round(CAT_FN[1]/CAT_FN[0], 3)*100, 1)}% of FNs in CAT is due to Limitation of Class-A methods.")
print(f"{round(round(Purity_FN[1] / Purity_FN[0], 3)*100, 1)}% of FNs in Purity is due to Limitation of Class-B methods.")
print(f"{round(round(CIT_FN[1]/CIT_FN[0], 3)*100, 1)}% of FNs in CIT is due to Limitation of Class-B methods.")
print(f"{round(round(PatInv_FN[1]/PatInv_FN[0], 3)*100, 1)}% of FNs in PatInv is due to Limitation of Class-C methods.")
print(f"{round(round(SIT_FN2[1]/SIT_FN2[0], 3)*100, 1)}% of FNs in SIT is due to Limitation of Structure-based methods.")
print(f"{round(round(CIT_FN2[1]/CIT_FN2[0], 3)*100, 1)}% of FNs in CIT is due to Limitation of Structure-based methods.")
print(f"{round(round(CAT_FP2[1]/CAT_FP2[0], 3)*100, 1)}% of FPs in CAT is due to Limitation of Text-based methods.")
print(f"{round(round(RTI_FP2[1]/RTI_FP2[0], 3)*100, 1)}% of FPs in RTI is due to Limitation of Text-based methods.")


print(f"{round(round(SIT_FP[1]/SIT_FP[0], 3)*100, 1)}% of FPs in SIT is due to Limitation of Class-A methods.", file=f_log)
print(f"{round(round(CAT_FP[1]/CAT_FP[0], 3)*100, 1)}% of FPs in CAT is due to Limitation of Class-A methods.", file=f_log)
print(f"{round(round(SIT_FN[1]/SIT_FN[0], 3)*100, 1)}% of FNs in SIT is due to Limitation of Class-A methods.", file=f_log)
print(f"{round(round(CAT_FN[1]/CAT_FN[0], 3)*100, 1)}% of FNs in CAT is due to Limitation of Class-A methods.", file=f_log)
print(f"{round(round(Purity_FN[1] / Purity_FN[0], 3)*100, 1)}% of FNs in Purity is due to Limitation of Class-B methods.", file=f_log)
print(f"{round(round(CIT_FN[1]/CIT_FN[0], 3)*100, 1)}% of FNs in CIT is due to Limitation of Class-B methods.", file=f_log)
print(f"{round(round(PatInv_FN[1]/PatInv_FN[0], 3)*100, 1)}% of FNs in PatInv is due to Limitation of Class-C methods.", file=f_log)
print(f"{round(round(SIT_FN2[1]/SIT_FN2[0], 3)*100, 1)}% of FNs in SIT is due to Limitation of Structure-based methods.", file=f_log)
print(f"{round(round(CIT_FN2[1]/CIT_FN2[0], 3)*100, 1)}% of FNs in CIT is due to Limitation of Structure-based methods.", file=f_log)
print(f"{round(round(CAT_FP2[1]/CAT_FP2[0], 3)*100, 1)}% of FPs in CAT is due to Limitation of Text-based methods.", file=f_log)
print(f"{round(round(RTI_FP2[1]/RTI_FP2[0], 3)*100, 1)}% of FPs in RTI is due to Limitation of Text-based methods.", file=f_log)
