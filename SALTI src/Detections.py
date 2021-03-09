class Detections():
    def __init__(self, boxes=list, classes=list, confidences=list):
        self.boxes=boxes
        self.classes=classes
        self.confidences=confidences

    def copy(self):
        return Detections(self.boxes.copy(), self.classes.copy(),self.confidences.copy())

    def append(self,other):
        out = self.copy()
        out.boxes       += other.boxes
        out.classes     += other.classes
        out.confidences += other.confidences
        return out

def add_classes(a, b):
    return Detections(a.boxes + b.boxes, a.classes + b.classes, a.confidences + b.confidences)

def test_detections():
    a=Detections([1,2],[3,4],[5,6])
    b=Detections([11,122],[13,14],[15,16])
    L_before = len(a.confidences)
    c=a.append(b)
    print(a)
    print(c)
    assert(len(a.confidences)==L_before)


test_detections()
