class Detections():
    def __init__(self, boxes, classes , confidences):
        self.boxes=boxes
        self.classes=classes
        self.confidences=confidences

    def append(self,other_detection):
        self.boxes=self.boxes+other_detection.boxes
        self.classes=self.classes+other_detection.classes
        self.confidences=self.confidences+other_detection.confidences
        return self

def test_detections():
    a=Detections([1,2],[3,4],[5,6])
    b=Detections([11,122],[13,14],[15,16])
    c=a.append(b)
    print(c)
