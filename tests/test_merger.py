import SALTI

def test_merger():
    b=SALTI.Detections([[1,2,3,4],[1,2,3,4]],[1,1],[0.1,0.7])
    a=SALTI.Merger(0.5,0.5)
    c=a.NMS(b)
    print(c.boxes)
    print(c.classes)
    print(c.confidences)
    print('length'+str(len(c.classes)))
    assert(len(c.classes)==1)

test_merger()
