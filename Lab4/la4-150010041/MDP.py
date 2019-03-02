import sys
import copy
filename = sys.argv[1]

file = open(filename, 'r')
text = file.read().split('\n')
del text[-1]

numStates = int(text[0].split(' ')[1])
numActions = int(text[1].split(' ')[1])

start = int(text[2].split(' ')[1])
end = int(text[3].split(' ')[1])

gamma = 0

data = []

for s in range(0, numStates):
    data.append([])
    for a in range(0, numActions):
        data[s].append([])

i = 4
while(True):
    row = text[i].split(' ')
    if row[0] == 'transitions':
        s = int(row[1])
        a = int(row[2])
        sPrime = int(row[3])
        r = float(row[4])
        p = float(row[5])

        data[s][a].append([sPrime, r, p])
    elif row[0] == 'discount':
        gamma = float(row[2])
        break
    i+=1


def valueIteration(value):
	global end, numStates, numActions, data
	value_updated = copy.deepcopy(value)
	for state in range(numStates):
		if state == end:
			continue
		maxValue = 0
		maxAction = 0
		for action in range(numActions):
			SAvalue = 0
			for sPrime in data[state][action]:
				SAvalue += sPrime[2] * (sPrime[1] + gamma*value[0][sPrime[0]])
			if SAvalue > maxValue:
				maxValue = SAvalue
				maxAction = action
		value_updated[0][state] = maxValue
		value_updated[1][state] = maxAction

	return value_updated

stateValues = [ [0.0] * numStates, range(numStates)]
stateValues[1][end] = -1
iterations = 0

while True:
	iterations+=1
	newStateValues = valueIteration(stateValues)
	flag = 1
	for i in range(numStates):
		if abs(newStateValues[0][i] - stateValues[0][i]) > 1E-16:
			flag = 0

	stateValues = newStateValues
	if flag == 1:
		break


for i in zip(stateValues[0], stateValues[1]):
	print i[0], i[1]
print 'iterations', iterations








# print numStates
# print numActions

# print start
# print end

# print data

# print gamma