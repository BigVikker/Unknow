import os
directory = 'Img'
listFileName = []
sum = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):
        listFileName.append(filename)
        sum += 1
listFileName.sort()
print(sum)
with open("Result/numberPlateResult.txt") as fp:
    Lines = fp.readlines() 
    for line in Lines: 
        sum += 1
        print(line.strip())
