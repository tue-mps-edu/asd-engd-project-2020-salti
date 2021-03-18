from Detector import Detector
from DataLoader import DataLoader
from Preprocesser import Preprocessor
from Visualizer import ProgressWindow
from Merger import Merger
from DataExporter import DataExporter


def SALTI(config):

    output_size = [config['int_output_x_size'], config['int_output_y_size']]

    path_rgb = config['str_dir_rgb']
    path_thermal = config['str_dir_thermal']
    path_output = config['str_dir_output']

    data = DataLoader(path_rgb,path_thermal,debug=False)

    # Preprocessor for RGB image with enhancing = False
    pp_c = Preprocessor(output_size=output_size, enhancing=False)

    # Preprocessor for thermal image with enhancing = True
    pp_t = Preprocessor(output_size=output_size, enhancing=config['bln_dofilter'])

    net_c = Detector('RGB')
    net_t = Detector('Thermal', config['dbl_thermal_conf'], config['dbl_thermal_nms'])
    RGB_classNames = net_c.get_classes() #Should be improved and retrieved through a getter (make it private attribute)

    # Initialize merger
    merge_c   = Merger( config['dbl_rgb_conf'],     config['dbl_rgb_nms'])
    merge_t   = Merger( config['dbl_thermal_conf'], config['dbl_thermal_nms'])
    merge_all = Merger( 0.2,                         config['dbl_merge_nms'])

    label_type = config['str_label']
    exporter = DataExporter(label_type, path_output , RGB_classNames, config['bln_validationcopy'])


    for file_name, file_ext, img_c, img_t in data:

        # img_plt is defined for visualizing the final thermal image without any pre-processing
        #img_plt = img_t.copy() # Wrong. this is not resized.

        # Output size added as an additional argument to define condition for resizing
        img_c_out, img_c = pp_c.process(img_c)
        img_t_out, img_t = pp_t.process(img_t)

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c.copy()))
        det_t = merge_t.NMS(net_t.detect(img_t.copy()))
        det_m = det_c+det_t
        det_m = merge_all.NMS(det_m)

        # Visualize
        #V = Visualize_all(img_c.copy(), img_plt)
        #V.print(RGB_classNames, det_c, det_t, det_m)
        
        # Progress window
        V = ProgressWindow(img_c, img_t, img_t_out, det_c, det_t, det_m, RGB_classNames, data.progress, config)

        # Export data
        exporter.export(output_size,file_name, file_ext,det_m, img_t_out)



