#from Detector import *

from Detector import YOLOv3_320, YoloJoeHeller
from Dataloader import Dataloader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
from Merger import Merger
from DataExporter import DataExporter
import cv2
from Visualizer import Visualizer
from Detections import Detections, add_classes


def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'].get(), outputs['y_size'].get()]

    path_rgb = dirs['rgb'].get()
    path_thermal = dirs['thermal'].get()
    path_output = dirs['output'].get()
    data = Dataloader(path_rgb,path_thermal,debug=True)

    fake_classes = ['car' for x in range(0,64)]
    label_type = 'PascalVOC'
    exporter = DataExporter(label_type,dirs['output'].get(),fake_classes)

    do_resize = (data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)

    # Initialize networks
    net_c, net_t = YOLOv3_320(), YoloJoeHeller()

    # Initialize merger
    merge_c   = Merger( thres['rgb_conf'].get(),     thres['rgb_nms'].get())
    merge_t   = Merger( thres['thermal_conf'].get(), thres['thermal_nms'].get())
    merge_all = Merger( 0.0,                         thres['merge_nms'].get())

    for file_name, file_ext, img_c, img_t in data:
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c))
        det_t = net_t.detect(img_t)
        det_m = det_c+det_t
        det_m = merge_all.NMS(det_m)

        # Visualize
        V = Visualize_all(img_c, img_t)
        V.print(fake_classes,det_c,det_t,det_m)

        # Export data
        exporter.export(img_t.shape,file_name,det_m)


