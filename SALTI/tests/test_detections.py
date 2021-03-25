from Detections import *

def test_detections():
    a=Detections([1,2],[3,4],[5,6])
    b=Detections([11,122],[13,14],[15,16])
    L_before = len(a.confidences)
    c=a.append(b)
    assert(len(a.confidences)==L_before)

test_detections()
