#from Detector import *
from Detector import YOLOv3_320, YoloJoeHeller
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
        det_c = net_c.detect(img_c)
        det_cf = merge_c.NMS(det_c)
        det_t = net_t.detect(img_t)
        # THERMAL NMS IS ALREADY IN DETECTOR, HERE IS DUPLICATE
        #det_tf = merge_t.NMS(det_t)
        det_tf = det_t
        det_m = merge_all.NMS(det_cf.append(det_t))

        # Visualize
        fake_classes = ['car' for x in range(0,4)]
        V = Visualize_all(img_c, img_t)
        V.print(fake_classes,det_cf,det_tf,det_m)



