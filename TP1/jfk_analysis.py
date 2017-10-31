import csv

file = open("jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv")

numField=len(file.readline().split(";"))
for f in file:

        if numField == len(f.split(";")):
            print("ok  line:  " + f)
        else:
            print("error")






