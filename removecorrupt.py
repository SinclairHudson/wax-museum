from os import listdir
from PIL import Image
count = 0
for filename in listdir('./dataset/notlooking'):
    try:
        img = Image.open('./dataset/notlooking/' + filename)  # open the image file
        img.verify()  # verify that it is, in fact an image
    except (IOError) as e:
        count += 1
        print('Bad file:', filename)  # print out the names of corrupt files
print(count)