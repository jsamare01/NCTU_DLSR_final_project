import os
def walk_dir(dir,topdown=True):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(dir)
             for filename in files]
    return files
fpath="./clothes/clothes_test/images"
lpath="./clothes/clothes_test/labels"
yolo="/tmp/work/Yolo/darknet/clothes/clothes_test/Yolos"
filename=walk_dir(fpath)
#train="/tmp/work/darknet/clothes/train/cfg/train.txt"
test="./clothes/clothes_test/cfg/test.txt"
#ftrain = open(train, "w")
ftest = open(test, "w")
empty=0
counter=0
for f in filename:
    number=f.split('/')[-1]
    label=number.split('.')[0]+'.txt'
    imgfile=os.path.join(yolo,number)
    lfile=os.path.join(lpath,label)
    yololabel=os.path.join(yolo,label)
    movelabel="cp  "+lfile+" "+yolo
    moveimg="cp "+f+" "+yolo
    os.system(moveimg)
    os.system(movelabel)
    if(os.stat(lfile).st_size == 0):
        
        if(empty>500):
             continue
        empty+=1
        #if(empty%5==0):
            #ftest.write(imgfile)
            
            #ftest.write('\n')
            #counter+=1
    
    else:
        ftest.write(imgfile)
        ftest.write('\n')
        counter+=1
print(counter)
