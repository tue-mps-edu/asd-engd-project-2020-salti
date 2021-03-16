from Detector import Detector
from DataLoader import DataLoader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
from Merger import Merger
from DataExporter import DataExporter


def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'], outputs['y_size']]

    path_rgb = dirs['rgb']
    path_thermal = dirs['thermal']
    path_output = dirs['output']

    data = DataLoader(path_rgb,path_thermal,debug=True)

    do_resize = (data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)

    net_c = Detector('RGB')
    net_t = Detector('Thermal', thres['thermal_conf'], thres['thermal_nms'])
    RGB_classNames = net_c.get_classes() #Should be improved and retrieved through a getter (make it private attribute)

    # Initialize merger
    merge_c   = Merger( thres['rgb_conf'],     thres['rgb_nms'])
    merge_t   = Merger( thres['thermal_conf'], thres['thermal_nms'])
    merge_all = Merger( 0.0,                         thres['merge_nms'])

    label_type = outputs['label']
    exporter = DataExporter(label_type, path_output , RGB_classNames)


    for file_name, file_ext, img_c, img_t in data:
        img_c = pp.process(img_c, do_resize, False)
        img_t = pp.process(img_t, do_resize, config['bln_dofilter'])

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c))
        det_t = net_t.detect(img_t)
        det_m = det_c+det_t
        det_m = merge_all.NMS(det_m)

        # Visualize
        V = Visualize_all(img_c.copy(), img_t.copy())
        V.print(RGB_classNames,det_c,det_t,det_m)

        # Export data
        exporter.export(img_t.shape,file_name, file_ext,det_m, img_t)



