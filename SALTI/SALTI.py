from Detector import Detector
from DataLoader import DataLoader
from Preprocesser import Preprocessor
from Visualizer import Visualize_all
from Merger import Merger
from DataExporter import DataExporter


def SALTI(config):

    output_size = [config['int_x_size'], config['int_y_size']]

    path_rgb = config['str_dir_rgb']
    path_thermal = config['str_dir_thermal']
    path_output = config['str_dir_output']

    data = DataLoader(path_rgb,path_thermal,debug=True)

    do_resize = (data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)

    net_c = Detector('RGB')
    net_t = Detector('Thermal', config['dbl_thermal_conf'], config['dbl_thermal_nms'])
    RGB_classNames = net_c.get_classes() #Should be improved and retrieved through a getter (make it private attribute)

    # Initialize merger
    merge_c   = Merger( config['dbl_rgb_conf'],     config['dbl_rgb_nms'])
    merge_t   = Merger( config['dbl_thermal_conf'], config['dbl_thermal_nms'])
    merge_all = Merger( 0.0,                         config['dbl_merge_nms'])

    label_type = config['str_label']
    exporter = DataExporter(label_type, path_output , RGB_classNames)


    for file_name, file_ext, img_c, img_t in data:
        #img_output = img_t.copy()
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

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



