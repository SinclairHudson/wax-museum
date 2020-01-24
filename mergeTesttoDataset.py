from os import listdir
import shutil

count = 0
for filename in listdir('./testset/notlooking'):
    if count % 1 == 0:
        shutil.move('./testset/notlooking/'+filename, './dataset/notlooking/'+filename)

    count = count + 1

count = 0
for filename in listdir('./testset/looking'):
    if count % 1 == 0:
        shutil.move('./testset/looking/' + filename, './dataset/looking/' + filename)

    count = count + 1