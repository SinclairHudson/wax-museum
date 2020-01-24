from os import listdir
import shutil

count = 0
for filename in listdir('./dataset/notlooking'):
    if count % 7 == 0:
        shutil.move('./dataset/notlooking/'+filename, './testset/notlooking/'+filename)

    count = count + 1

count = 0
for filename in listdir('./dataset/looking'):
    if count % 7 == 0:
        shutil.move('./dataset/looking/' + filename, './testset/looking/' + filename)

    count = count + 1