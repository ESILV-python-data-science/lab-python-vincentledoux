import csv
import builtins

file = open("jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv")

page = 0
pageCount = 0
count = 0
pageMax = 0
pageMin = 100
countBug = 0
dictionary = {}
dictionary2 = {}
dictionaryDocumentType = {}
countype = 0
countype2 = 0

numField = len(file.readline().split(";"))
for f in file:

        tab = f.split(";")

        try:
            count = count + 1
            page = int(tab[11])
            pageCount = pageCount + page

            if page > pageMax:
                pageMax = page
            if count == 1:
                pageMin = page
            if page < pageMin:
                pageMin = page

            if numField == len(tab):
                print("ok  line:  " + f)
            else:
                print("error")

            if str(tab[6]) not in dictionary.keys():
                dictionary.update({str(tab[6]): 1})
            if str(tab[4]) not in dictionary2.keys():
                dictionary2.update({str(tab[4]): 1})

            if str(tab[6]) in dictionary:
                dictionary[str(tab[6])] += 1

            if str(tab[4]) in dictionary2:
                dictionary2[str(tab[4])] += 1

        except ValueError:
            countBug += 1


print(pageCount/count)
print("max : " + pageMax.__str__() + " min : " + pageMin.__str__())
print("number of types of documents is : " + str(dictionary.__len__()))
print("number of agencies is : " + str(dictionary2.__len__()))

for v, i in dictionary.items():
    print(str(v), str(i))

for v, i in dictionary2.items():
    print(str(v), str(i))








