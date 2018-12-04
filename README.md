# CarND-Capstone-TrainingImgs
### 1.Images(2018-12-01):
#### SimulatorTrack1_Classified_Imgs
RED: 706   
YELLOW: 66   
GREEN: 489  
UNKNOWN: 288   
#### RealCarTrack_Unclassified_Imgs
243  
#### FromNet_Traffic_Light_Imgs , images from UdaCity "Intro to Self-Driving Cars" course
training: 1187  
test: 297   
   
   
   
### 2. a bit of py scripts(2018-12-03) 
#### Functions in train.py
get_traindata() # read image from disk into 2 python lists.  
train() # a cnn to test data   
   
need:  
tensorflow 1.3.0  
sklearn  

usage:  
```
python train.py  
```
   
result:  
...  
112  train_accuracy:  0.78   
113  train_accuracy:  0.74  
114  train_accuracy:  0.76  
115  train_accuracy:  0.78  
116  train_accuracy:  0.86  
117  train_accuracy:  0.84  
118  train_accuracy:  0.82  
119  train_accuracy:  0.88  
120  train_accuracy:  0.82  
121 -----Test_accuracy:  [0.85483873]  
   
   
### 3. Manual annotation(2018-12-03)
####  Add 60 simulator image annotations manually (use labelimg,  20 XMLs per traffic light color) for test, if test is ok, label image else. 
   
label file folder:   
./SimulatorTrack1_Classified_Imgs/GREEN_label   
./SimulatorTrack1_Classified_Imgs/RED_label   
./SimulatorTrack1_Classified_Imgs/YELLOW_label   
   
XML example (NOTE: 4th line in XML has a full path, when use those files, maybe need to modify it):    
```
<annotation>
        <folder>RED</folder>
        <filename>IMG_15437218315.jpg</filename>
        <path>/jixj/term3/p015/SimulatorTrack1_Classified_Imgs/RED/IMG_15437218315.jpg</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>800</width>
                <height>600</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>126</xmin>
                        <ymin>256</ymin>
                        <xmax>181</xmax>
                        <ymax>382</ymax>
                </bndbox>
        </object>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>374</xmin>
                        <ymin>259</ymin>
                        <xmax>432</xmax>
                        <ymax>386</ymax>
                </bndbox>
        </object>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>624</xmin>
                        <ymin>263</ymin>
                        <xmax>688</xmax>
                        <ymax>390</ymax>
                </bndbox>
        </object>
</annotation>

```
predefined_classes.txt is a classes file for labelimg. 
