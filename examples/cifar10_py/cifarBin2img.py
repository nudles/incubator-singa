
import sys, os

#all the settings

SINGA_ROOT=os.path.join(os.path.dirname(__file__),'../','../')
sys.path.append(os.path.join(SINGA_ROOT,'tool','python'))
from singa.model import * 
from singa.utils import imgtool
from PIL import Image

label_map=dict()

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def test():
    im = Image.open("dog.jpg").convert("RGB")

    byteArray=imgtool.toBin(im,(32,32))
    im2 = imgtool.toImg(byteArray,(32,32))

    im2.save("dog2.jpg", "JPEG")
    

def getLabelMap(path):
    d = unpickle(path)
    index=0
    for line in d["label_names"]:
        print index,line
        label_map[index]=line
        index+=1
    return

def generateImage(input_path,output_path,random):
    dict=unpickle(input_path)
    data=dict["data"]
    labels=dict["labels"]
    index=0
    for d in data:
        im = imgtool.toImg(data[index],(32,32))
        temp_folder=os.path.join(output_path,label_map[labels[index]])
        try:
            os.stat(temp_folder)
        except:
            os.makedirs(temp_folder)
        im.save(os.path.join(temp_folder,random+"_"+str(index)+".jpg"),"JPEG") 
        index+=1
    #print labels

getLabelMap("data/batches.meta")
#generateImage("data/data_batch_1", "data/output","1")
#generateImage("data/data_batch_2", "data/output","2")
#generateImage("data/data_batch_3", "data/output","3")
#generateImage("data/data_batch_4", "data/output","4")
#generateImage("data/data_batch_5", "data/output","5")
#generateImage("data/test_batch", "data/output","6")






