import json

number = open('csv/Book1.csv', 'r').readlines()
b = []
for elm in number:
    tmp = elm.split(',')
    tmp = [tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5][:-2]]
    if tmp[1] != '':
        b.append({tmp[0]: [tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]]})

with open('list_withbndbx.json', 'w') as wr:
    json.dump(b, wr)
