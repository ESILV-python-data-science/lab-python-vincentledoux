import csv

file = open("jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv")

page = 0
pageCount = 0
count = 0
pageMax = 0
pageMin = 0

numField = len(file.readline().split(";"))
for f in file:
        count = count + 1
        tab = f.split(";")
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

print(pageCount/count)
print("max : " + pageMax + " min : " + pageMin)









