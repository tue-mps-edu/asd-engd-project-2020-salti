#from Detector import *
from Dataloader import Dataloader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
import cv2

def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'].get(), outputs['y_size'].get()]

    path_rgb = dirs['rgb'].get()
    path_thermal = dirs['thermal'].get()
    data = Dataloader(path_rgb,path_thermal)
    do_resize = ( data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)


    #net_C, net_T = YOLOv3_320(), YoloJoeHeller()

    for img_c, img_t in data:
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

        cv2.imshow('RGB',img_c)
        cv2.imshow('Thermal',img_t)
        cv2.waitKey(100)
        Visualize_all(img_c, img_t)
        print('dbstop')

        V
        #det_c = net_c.detect(img_c)
        #det_t = net_t.detect(img_t)





