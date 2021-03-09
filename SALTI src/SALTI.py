#from Detector import *

from Detector import YOLOv3_320, YoloJoeHeller
from Dataloader import Dataloader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
from Visualizer import Visualizer
from Detections import Detections, add_classes
from Merger import Merger
import cv2

def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'].get(), outputs['y_size'].get()]

    path_rgb = dirs['rgb'].get()
    path_thermal = dirs['thermal'].get()
    data = Dataloader(path_rgb,path_thermal)
    do_resize = ( data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)

    # Initialize networks
    net_c, net_t = YOLOv3_320(), YoloJoeHeller()

    # Initialize merger
    merge_c   = Merger( thres['rgb_conf'].get(),     thres['rgb_nms'].get())
    merge_t   = Merger( thres['thermal_conf'].get(), thres['thermal_nms'].get())
    merge_all = Merger( 0.0,                         thres['merge_nms'].get())

    for img_c, img_t in data:
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c))
        det_t = net_t.detect(img_t)
        L1 = len(det_c.confidences)
        L2 = len(det_t.confidences)

        det_m = add_classes(det_c, det_t)
        assert(L1==len(det_c.confidences))
        assert(L2==len(det_t.confidences))

        # Here it goes wrong. It has to do with class copies.
        # Solve this if you want to really learn python ;)
        det_m = det_c.append(det_t)
        assert(L1==len(det_c.confidences))
        assert(L2==len(det_t.confidences))



        fake_classes = ['car' for x in range(0,4)]

        # Separate printing
        V_C = Visualizer(img_c)
        V_C.print_annotated_image('RGB',fake_classes,det_c)
        V_T = Visualizer(img_t)
        V_T.print_annotated_image('Thermal',fake_classes,det_t)
        #V_M = Visualizer(img_t)
        #V_M.print_annotated_image('Thermal',fake_classes,det_m)
        cv2.waitKey(1000)



        # Visualize

        #V = Visualize_all(img_c, img_t)
        #V.print(fake_classes,det_cf,det_tf,det_m)



