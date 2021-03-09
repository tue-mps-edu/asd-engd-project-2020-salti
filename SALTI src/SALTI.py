#from Detector import *
from Detector import YOLOv3_320
from Dataloader import Dataloader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
from Visualizer import Visualizer
from Detections import Detections
from Merger import Merger
import cv2

def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'].get(), outputs['y_size'].get()]

    path_rgb = dirs['rgb'].get()
    path_thermal = dirs['thermal'].get()
    data = Dataloader(path_rgb,path_thermal)
    do_resize = ( data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)


    net_c = YOLOv3_320()
    #net_t = YoloJoeHeller()

    for img_t, img_c in data:
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

        # Do detections
        Merger(0.5, 0.5)
        det_c = net_c.detect(img_c)
        #det_t = net_t.detect(img_t)
        det_t, = Detections()

        det_m = Merger(0.5, 0.5)


        # Visualize
        V = Visualizer(img_c)
        V.print_annotated_image('RGB',['car' for x in range(0,4)],det_c)

        #Visualize_all(img_c, img_t)
        print('dbstop')

        #det_c = net_c.detect(img_c)
        #det_t = net_t.detect(img_t)




