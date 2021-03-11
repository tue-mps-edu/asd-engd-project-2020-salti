class Detections():
    def __init__(self, boxes=list, classes=list, confidences=list):
        self.boxes = boxes
        self.classes = classes
        self.confidences = confidences

    def __copy__(self):
        # Overload copy operator
        return Detections(self.boxes.copy(), self.classes.copy(),self.confidences.copy())

    def append(self,other):
        out = self.__copy__()
        out.boxes += other.boxes
        out.classes += other.classes
        out.confidences += other.confidences
        return out

    def __add__(self, other):
        # Overload + operator
        out = self.__copy__()
        out.boxes += other.boxes
        out.classes += other.classes
        out.confidences += other.confidences
        return out
