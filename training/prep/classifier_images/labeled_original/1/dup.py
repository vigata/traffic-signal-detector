from shutil import copyfile

origs = [ 1000+i for i in range(5) ]

for i in range (15):
    for orig in origs:
        copyfile("{}.png".format(orig), "{}_{}.png".format(i, orig))

 
