class Detections():
    def __init__(self, boxes=list, classes=list, confidences=list):
        '''
        Class that is used as interface for the Detections.

        Arguments:
            (list) list containing lists of BBOX coordinates
            (list) class ids
            (list) confidences
        '''
        self.boxes = boxes
        self.classes = classes
        self.confidences = confidences

    def __copy__(self):
        '''Copy constructor'''
        return Detections(self.boxes.copy(), self.classes.copy(),self.confidences.copy())

    def append(self,other):
        '''Append all classes inside the Detections'''
        out = self.__copy__()
        out.boxes += other.boxes
        out.classes += other.classes
        out.confidences += other.confidences
        return out

    def __add__(self, other):
        '''Overload + operator'''
        out = self.__copy__()
        out.boxes += other.boxes
        out.classes += other.classes
        out.confidences += other.confidences
        return out
