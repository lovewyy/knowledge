# 01bags

def findBag(w, v, c):
    n = len(w)
    r = [ [ 0 for j in range(c+1)] for i in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, c+1):
            if j < w[i-1]:
                r[i][j] = r[i-1][j]
            else:
                r[i][j] = max(r[i-1][j], r[i-1][j-w[i-1]] + v[i-1])
    '''
    for i in range(n+1):
        for j in range(c+1):
            print(r[i][j], end = ' ')
        print()
    '''
    (i, j) = (n, c)
    result = []
    while i > 0 and j > 0:
        if r[i][j] == r[i-1][j]:
            i -= 1
        elif r[i][j] == r[i-1][j-w[i-1]] + v[i-1]:
            result.append(w[i-1])
            j -= w[i-1]
            i -= 1
    return r[-1][-1], result[::-1]

path = "C:\\Users\\l50002801\\Desktop\\Test-Data-of-01-knapsack-problem--master\\test\\"
for i in range(10):
    pathin = path + 'beibao' + str(i) + '.in'
    pathout = path + 'beibao' + str(i) + '.out'
    # print(pathin)
    test_list = []
    w = []
    v = []
    result0 = -1
    result1 = []
    with open(pathin, 'r') as fr:
        lines = fr.read().splitlines()
        for line in lines:
            line = line.split(' ')
            w.append(int(line[0]))
            v.append(int(line[1]))
        c = v[0]
        w = w[1:]
        v = v[1:]
        # print(c)
        # print(w)
        # print(v)
        result = findBag(w, v, c)
        result0 = result[0]
        result1 = result[1]
        
    result2 = -2
    with open(pathout, 'r') as fr:
        lines = fr.read().splitlines()
        result2 = int(lines[0])
    
    # print(result1)
    print(result0 == result2)
