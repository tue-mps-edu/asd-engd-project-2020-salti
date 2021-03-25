from Merger import *

def test_merger():
    b=Detections([[1,2,3,4],[1,2,3,4]],[1,1],[0.1,0.7])
    a=Merger(0.5,0.5)
    c=a.NMS(b)
    print(c.boxes)
    print(c.classes)
    print(c.confidences)
    print('length'+str(len(c.classes)))
    assert(len(c.classes)==1)

