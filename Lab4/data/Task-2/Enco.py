import sys
import copy
filename = sys.argv[1]

file = open(filename, 'r')
text = file.read().split('\n')
del text[-1]

grid = [ row.split(' ') for row in text]
rows = len(grid)
columns = len(grid[0])
numStates = rows * columns

start = 0
end = 0

for i in range(1, rows):
	for j in range(1, columns):
		if grid[i][j] == '2':
			start = rows*i+j
		if grid[i][j] == '3':
			end = rows*i+j

# North - 0, East - 1 , South - 2, West - 3

print "numStates", rows*columns
print "numActions", 4

print "start",start

print "end",end

states = {}


for i in range(1, rows-1):
	for j in range(1, columns-1):
		if grid[i][j] == '0':
			if grid[i+1][j] != '1':
				print 'transitions', rows * i + j, 2, rows * (i+1) + j, -1.0, 1.0
			if grid[i-1][j] != '1':
				print 'transitions', rows * i + j, 0, rows * (i-1) + j, -1.0, 1.0
			if grid[i][j+1] != '1':
				print 'transitions', rows * i + j, 1, rows * i + (j+1), -1.0, 1.0
			if grid[i][j-1] != '1':
				print 'transitions', rows * i + j, 3, rows * i + (j-1), -1.0, 1.0

for i in range(1, rows-1):
	for j in range(1, columns-1):
		if grid[i][j] == '0':
			if grid[i+1][j] == '1':
				print 'transitions', rows * i + j, 2, rows * i + j, -100.0, 1.0
			if grid[i-1][j] == '1':
				print 'transitions', rows * i + j, 0, rows * i + j, -100.0, 1.0
			if grid[i][j+1] == '1':
				print 'transitions', rows * i + j, 1, rows * i + j, -100.0, 1.0
			if grid[i][j-1] == '1':
				print 'transitions', rows * i + j, 3, rows * i + j, -100.0, 1.0

for i in range(0, rows):
	for j in range(0, columns):
		if grid[i][j] == '1':
			print 'transitions', rows * i + j, 2, rows * i + j, -100.0, 1.0
			print 'transitions', rows * i + j, 0, rows * i + j, -100.0, 1.0
			print 'transitions', rows * i + j, 1, rows * i + j, -100.0, 1.0
			print 'transitions', rows * i + j, 3, rows * i + j, -100.0, 1.0

for i in range(1, rows-1):
	for j in range(1, columns-1):
		if grid[i][j] == '2':
			if grid[i+1][j] != '1':
				print 'transitions', rows * i + j, 2, rows * (i+1) + j, -1.0, 1.0
			if grid[i-1][j] != '1':
				print 'transitions', rows * i + j, 0, rows * (i-1) + j, -1.0, 1.0
			if grid[i][j+1] != '1':
				print 'transitions', rows * i + j, 1, rows * i + (j+1), -1.0, 1.0
			if grid[i][j-1] != '1':
				print 'transitions', rows * i + j, 3, rows * i + (j-1), -1.0, 1.0

for i in range(1, rows-1):
	for j in range(1, columns-1):
		if grid[i][j] == '2':
			if grid[i+1][j] == '1':
				print 'transitions', rows * i + j, 2, rows * i + j, -100.0, 1.0
			if grid[i-1][j] == '1':
				print 'transitions', rows * i + j, 0, rows * i + j, -100.0, 1.0
			if grid[i][j+1] == '1':
				print 'transitions', rows * i + j, 1, rows * i + j, -100.0, 1.0
			if grid[i][j-1] == '1':
				print 'transitions', rows * i + j, 3, rows * i + j, -100.0, 1.0

print "discount ",1