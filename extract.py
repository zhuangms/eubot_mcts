import csv
from sys import flags

s = open('record_wie1.csv', 'r', newline='')
lines = csv.reader(s)

cnt = 0
cur = ''
rStateNum, rStateIE = [], []
tStateNum, tStateIE = [], []

rNumMax, rIEMax = 0, 0
tNumMax, tIEMax = 0, 0
for line in lines:
    if cnt > 0 and cnt < 2:
        if cur == 'random' and len(rStateIE) < 28:
            rStateIE.append(float(line[0]))
            for i in range(len(line)):
                if i >= 2:
                    rIEMax = max(rIEMax, float(line[i]))
        
        if cur == 'trout' and len(tStateIE) < 28:
            tStateIE.append(float(line[0]))
            for i in range(len(line)):
                if i >= 2:
                    tIEMax = max(tIEMax, float(line[i]))
        cnt += 1
    else:
        cnt = 0

    if line == '' or len(line) == 0:
        continue

    if line[1] == 'random' and len(rStateNum) < 28:
        cnt += 1
        cur = 'random'
        rStateNum.append(float(line[0]))
        for i in range(len(line)):
            if i >= 2:
                rNumMax = max(rNumMax, int(line[i]))

    if line[1] == 'trout' and len(tStateNum) < 28:
        cnt += 1
        cur = 'trout'
        tStateNum.append(float(line[0]))
        for i in range(len(line)):
            if i >= 2:
                tNumMax = max(tNumMax, int(line[i]))

print('random_state_avg: ', sum(rStateNum) / len(rStateNum))
print('random_ie_avg: ', sum(rStateIE) / len(rStateIE))
print('random_state_max: ', rNumMax)
print('random_ie_max: ', rIEMax)

print('trout_state_avg: ', sum(tStateNum) / len(tStateNum))
print('trout_ie_avg: ', sum(tStateIE) / len(tStateIE))
print('trout_state_max: ', tNumMax)
print('trout_ie_max: ', tIEMax)
# print(sum(sStateNum) / len(sStateNum))
# print(sum(sStateIE) / len(sStateIE))
# print(sum(sStateCorrect)/ len(sStateCorrect))
# print(sum(sPickRight) / len(sPickRight))