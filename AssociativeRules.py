from collections import Counter

import numpy as np
import itertools
import pandas as pd

def support(A, B = None, data = None):
    res = 0
    for i in data:
        flag = True
        for j in A:
            flag = j in i
            if not flag: break

        if flag and B != None:
            for j in B:
                flag = j in i
                if not flag: break

        if flag: res += 1
    return res / len(data)

def confidence(A, B, data):
    resAB = 0
    resA = 0
    for i in data:
        flag = True
        for j in A:
            flag = j in i
            if not flag: break

        if flag:
            for j in B:
                flag = j in i
                if not flag: break
        if flag: resAB += 1

        flag = True
        for j in A:
            flag = j in i
            if not flag: break
        if flag: resA += 1

    if resA == 0:
        return resAB
    else:
        return resAB / resA

def lift(A, B, data):
    S = support(B, data = data)
    C = confidence(A,B,data)
    if S == 0:
        return C
    else:
        return  C / S

def leverage(A, B, data):
    return support(A,B, data) - support(A,data=data) * support(B,data=data)

def count(A, data):
    res = 0
    for i in data:
        flag = True
        for j in A:
            flag = j in i
            if not flag: break

        if flag: res += 1
    return res

def all_in(A, B):
    flag = True
    for j in A:
        flag = j in B
        if not flag: break
    return flag

def isnt_in_(A: tuple, list_B: list):
    if list_B != []:
        for B in list_B:
            if A in itertools.permutations(B, len(B)):
                return False
    return True

def connect(arg):
    if type(arg[0]) == str:
        return [i for i in itertools.combinations(arg, 2)]
    res = []
    for i in arg:
        for j in arg:
            if i != j:
                if all_in(i[:-1], j[:-1]):
                    candidate = i + (j[-1],)
                    if isnt_in_(candidate, res):
                        res.append(candidate)
    return res

def antimonoton_list(list_,lim,data):
    if type(list_) == set: return list(list_)
    res = []
    for j in list_:
        flag = True
        if len(j) > 2:
            set_items = itertools.combinations(j, len(j) - 1)
            for i in set_items:
                if count(i, data) < lim:
                    flag = False
                    break
        else:
            if count(j, data) < lim:
                flag = False

        if flag: res.append(j)
    return res

def create_consequence(arg):
    res = []
    for i in arg:
        for j in range(len(i)):
            temp = list(i)
            temp.pop(j)
            # print(temp, ' --> ', i[j])
            res.append([tuple(temp),i[j]])
    return res

def count_all(reas,cons, data):
    res = []
    S = support(reas,cons, data)
    res.append(S)
    C = confidence(reas,cons, data)
    res.append(C)
    res.append(leverage(reas,cons, data))
    res.append(lift(reas,cons, data))
    res.append(S*C)
    return res


#################
#apiori alg

def eval(data):
    set_items = set([item for sublist in data for item in sublist])
    combinations = itertools.permutations(set_items, 2)
    # [print(i) for i in combinations]

    res = pd.DataFrame(
        [[" ".join(i), leverage(list(i[:-1]), [i[-1]], data), lift(list(i[:-1]), [i[-1]], data)] for i in combinations],
        columns=['comb', 'lev', 'lift'])
    res = res.sort_values(by=['lev'], ascending=False)
    return res

def eval_Apriori(lim,data):

    dataApriori = pd.DataFrame([], columns=['reas','cons', 'sup', 'conf', 'lev', 'lift', 'SxC'])
    set_items = set([item for sublist in data for item in sublist])
    list_cur = antimonoton_list(set_items,lim,data)

    for k in range(len(set_items)):
        list_cur = connect(list_cur)
        if list_cur == []: break
        list_cur = antimonoton_list(list_cur,lim,data)
        cons = create_consequence(list_cur)
        for i in cons:
            j = i
            dataApriori.loc[len(dataApriori)] = j + count_all(list(j[0]), [j[1]], data)
    sorted = dataApriori.sort_values(by=['SxC'], ascending=False)
    return sorted


if __name__ == '__main__':
    data = open('data/dataForRules.txt', 'r').readlines()
    data = [i.replace('\n', '').lower().split(', ') for i in data]
    print(eval(data))
    print(eval_Apriori(4, data))