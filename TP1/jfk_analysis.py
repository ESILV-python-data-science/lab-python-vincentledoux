import csv

file = open("jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv")

page = 0
pageCount = 0
count = 0
pageMax = 0
pageMin = 100
countNoPage = 0;

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
        except ValueError:

            countNoPage += 1


print(pageCount/count)
print("max : " + pageMax.__str__() + " min : " + pageMin.__str__())
print("Count no page : " + countNoPage.__str__())









