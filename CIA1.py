import numpy as np

#S1 - row
#S2 - columns

MATCH = 5
MISMATCH = -4

MAX = -1

def isCalculatable(visitedMtrix, row, column):
    if (visitedMtrix[row-1][column-1] == 1 and visitedMtrix[row][column-1] == 1 and visitedMtrix[row-1][column] == 1):
        return False
    return True


def DP(alignmentMatrix, row, column, visitedMatrix, s1, s2):
    
    if s1[row-1] == s2[column-1]:
        alignmentMatrix[row][column] = alignmentMatrix[row - 1][column - 1] + MATCH
    else:
        alignmentMatrix[row][column] = max(alignmentMatrix[row][column - 1] + MISMATCH, alignmentMatrix[row - 1][column] + MISMATCH)
    
    visitedMatrix[row][column] = 1
    
    if row == 16 and column == 16:
        return
    
    
    else:
        try:
            if visitedMatrix[row + 1][column] == 0:
                while(isCalculatable(visitedMatrix, row+1, column)):
                    print("waiting")
                    pass
                DP(alignmentMatrix, row + 1, column, visitedMatrix, s1, s2)
        except IndexError:
            DP(alignmentMatrix, 1, column + 1, visitedMatrix, s1, s2)


def backTrack(AM, row, column, s1, s2):
    
    if (row == column == 0):
        return
    else:
        if AM[row][column] - AM[row-1][column-1] == MATCH:
            print(s1[row - 1], s2[column - 1])
            backTrack(AM, row-1, column - 1, s1, s2)
        elif AM[row][column] - AM[row-1][column] == MISMATCH:
            print(s1[row - 1], "_")
            backTrack(AM, row-1, column, s1, s2)
        elif AM[row][column] - AM[row][column-1] == MISMATCH:
            print("_", s2[column - 1])
            backTrack(AM, row, column - 1, s1, s2)

        
AM = [[0 for i in range(17)] for j in range(17)]
VM = [[0 for i in range(17)] for j in range(17)]

for i in range(17):
    VM[0][i] = 1
    VM[i][0] = 1

S1 = "AAAACCCCGGGGTTTT"
S2 = "TAAAACCCCGGGGTTT"

DP(AM, 1, 1, VM, S1, S2)

for i in range(len(AM)):
    print(AM[i])
    
    
MAX_ELEMENTS = []
INDICES = []
    
for h in range(4):
    M = -999
    maxIndex = [0,0]
    for i in range(1,17):
        for j in range(1,17):
            if M < AM[i][j] and AM[i][j] not in MAX_ELEMENTS:
                M = AM[i][j]
                maxIndex = [i,j]
    
    MAX_ELEMENTS += [M]
    INDICES += [maxIndex]

print(MAX_ELEMENTS)
print(INDICES)

for i in range(4):
    print("=========================")
    backTrack(AM, INDICES[i][0], INDICES[i][1], S1, S2)


