import csv

s = open('history.csv', 'r', newline='')
lines = csv.reader(s)

cnt = 0
cur = ''
rStateNum, rStateIE, rStateCorrect, rPickRight = [], [], [], []
oStateNum, oStateIE, oStateCorrect, oPickRight = [], [], [], []
sStateNum, sStateIE, sStateCorrect, sPickRight = [], [], [], []
for line in lines:
    if cnt > 0 and cnt < 4:
        if cnt == 1:
            if cur == 'random':
                rStateIE.append(float(line[0]))
            elif cur == 'Oracle':
                oStateIE.append(float(line[0]))
            elif cur == 'StrangeFish2':
                sStateIE.append(float(line[0]))
        elif cnt == 2:
            if cur == 'random':
                rStateCorrect.append(float(line[0]))
            elif cur == 'Oracle':
                oStateCorrect.append(float(line[0]))
            elif cur == 'StrangeFish2':
                sStateCorrect.append(float(line[0]))
        elif cnt == 3:
            if cur == 'random':
                rPickRight.append(float(line[0]))
            elif cur == 'Oracle':
                oPickRight.append(float(line[0]))
            elif cur == 'StrangeFish2':
                sPickRight.append(float(line[0]))
        # print(line)
        cnt += 1
    else:
        cnt = 0
    if line[1] == 'random':
        # print(line)
        cnt += 1
        cur = 'random'
        rStateNum.append(float(line[0]))

    if line[1] == 'Oracle':
        # print(line)
        cnt += 1
        cur = 'Oracle'
        oStateNum.append(float(line[0]))

    if line[1] == 'StrangeFish2':
        cnt += 1
        cur = 'StrangeFish2'
        sStateNum.append(float(line[0]))

# print(sum(rStateNum) / len(rStateNum))
# print(sum(rStateIE) / len(rStateIE))
# print(sum(rStateCorrect) / len(rStateCorrect))
# print(sum(rPickRight) / len(rPickRight))

# print(sum(oStateNum) / len(oStateNum))
# print(sum(oStateIE) / len(oStateIE))
# print(sum(oStateCorrect)/ len(oStateCorrect))
# print(sum(oPickRight) / len(oPickRight))

print(sum(sStateNum) / len(sStateNum))
print(sum(sStateIE) / len(sStateIE))
print(sum(sStateCorrect)/ len(sStateCorrect))
print(sum(sPickRight) / len(sPickRight))