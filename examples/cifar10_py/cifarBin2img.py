
import sys, os
from numpy.core.test_rational import numerator

SINGA_ROOT=os.path.join(os.path.dirname(__file__),'../','../')
sys.path.append(os.path.join(SINGA_ROOT,'tool','python'))
from singa.model import * 
from singa.utils import imgtool
from PIL import Image
import cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def test():
    '''
        test imgtool toBin and toImg
    '''
    im = Image.open("dog.jpg").convert("RGB")

    byteArray=imgtool.toBin(im,(32,32))
    im2 = imgtool.toImg(byteArray,(32,32))

    im2.save("dog2.jpg", "JPEG")
    

def getLabelMap(path):
    d = unpickle(path)
    label_map=dict()
    for index,line in numerator(d["label_names"]):
        print index,line
        label_map[index]=line
    return label_map

def generateImage(input_path,output_path,label_map,random):
    dict=unpickle(input_path)
    data=dict["data"]
    labels=dict["labels"]
    for index,d in numerator(data):
        im = imgtool.toImg(data[index],(32,32))
        temp_folder=os.path.join(output_path,label_map[labels[index]])
        try:
            os.stat(temp_folder)
        except:
            os.makedirs(temp_folder)
        im.save(os.path.join(temp_folder,random+"_"+str(index)+".jpg"),"JPEG") 
    #print labels

def main():
    label_map=getLabelMap("data/batches.meta")
    generateImage("data/data_batch_1", "data/output",label_map,"1")
    generateImage("data/data_batch_2", "data/output",label_map,"2")
    generateImage("data/data_batch_3", "data/output",label_map,"3") 
    generateImage("data/data_batch_4", "data/output",label_map,"4")
    generateImage("data/data_batch_5", "data/output",label_map,"5")
    generateImage("data/test_batch", "data/output",label_map,"6")
    
if __name__=='__main__':
    main()





