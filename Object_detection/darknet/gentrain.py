import os
def walk_dir(dir,topdown=True):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(dir)
             for filename in files]
    return files
fpath="/tmp/work/darknet/clothes/train/images"
lpath="/tmp/work/darknet/clothes/train/labels"
yolo="/tmp/work/darknet/clothes/train/Yolos"
filename=walk_dir(fpath)
train="/tmp/work/darknet/clothes/train/cfg/train.txt"
test="/tmp/work/darknet/clothes/train/cfg/test.txt"
ftrain = open(train, "w")
ftest = open(test, "w")
empty=0
counter=0
for f in filename:
    number=f.split('/')[-1]
    label=number.split('.')[0]+'.txt'
    imgfile=os.path.join(yolo,number)
    lfile=os.path.join(lpath,label)
    print(lfile)
    if(os.stat(lfile).st_size == 0):
        
        if(empty>500):
             continue
        empty+=1
        if(empty%5==0):
            ftrain.write(imgfile)
            
            ftrain.write('\n')
            #counter+=1
    
    else:
        ftrain.write(imgfile)
        ftrain.write('\n')
        counter+=1
print(counter)
