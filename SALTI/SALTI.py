from Detector import Detector
from DataLoader import DataLoader
from Preprocesser import Preprocessor
from Visualizer import ProgressWindow
from Merger import Merger
from DataExporter import DataExporter

def SALTI(config):
    '''
    This function contains the SALTI labeling pipeline.
    It calls all functions from loading unlabeled images up to saving the labeled images.

    Parameters in config (dictionary with standard Python variable type):
        Parameters as defined in config.ini (can be updated throught the GUI)
        - Directories of images
        - Thresholds for confidence and non-maximum supression
        - output settings (labeltype, imagesize, filtered image, validation copy)
        - filter settings

    Outputs (in output directory defined in 'config'):
        A subdirectory with date-time stamp that contains:
        - Thermal images (resized)
        - Image labels (YOLO/PascalVOC)
        - (optional) Image label duplicate
        - (optional) Filtered images
    '''

    output_size = [config['int_output_x_size'], config['int_output_y_size'],3]

    path_rgb = config['str_dir_rgb']
    path_thermal = config['str_dir_thermal']
    path_output = config['str_dir_output']

    # Initialize class that loads all images
    data = DataLoader(path_rgb,path_thermal,debug=False)
    # Preprocessor for RGB image with enhancing = False
    pp_c = Preprocessor(output_size=output_size, enhancing=False)
    # Preprocessor for thermal image with enhancing = True
    pp_t = Preprocessor(output_size=output_size, enhancing=config['bln_dofilter'])

    # Initialize Object Detection classes
    net_c = Detector('RGB',config['dbl_thermal_conf'])
    net_t = Detector('Thermal',config['dbl_thermal_conf'])
    RGB_classNames = net_c.get_classes()

    # Initialize merger class for merging detections
    merge_c   = Merger( config['dbl_rgb_conf'],     config['dbl_rgb_nms'])
    merge_t   = Merger( config['dbl_thermal_conf'], config['dbl_thermal_nms'])
    merge_all = Merger( 0.0,                        config['dbl_merge_nms'])

    # Class for exporting the images & detections
    exporter = DataExporter(config, path_output, RGB_classNames)

    for file_name, file_ext, img_c, img_t in data:

        # Output size added as an additional argument to define condition for resizing
        img_c_out, img_c = pp_c.process(img_c)
        img_t_out, img_t = pp_t.process(img_t)

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c.copy()))
        det_t = merge_t.NMS(net_t.detect(img_t.copy()))
        det_m = det_c+det_t
        det_m = merge_all.NMS(det_m)

        # Progress window
        V = ProgressWindow(img_c, img_t, img_t_out, det_c, det_t, det_m, file_name+file_ext, RGB_classNames, data.progress, config)

        # Export data
        exporter.export(output_size,file_name, file_ext,det_m, img_t_out, img_t, config)

    print('Labeling completed.')
