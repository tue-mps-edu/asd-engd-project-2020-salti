from . import Detector as det
from . import DataLoader as dl
from . import Preprocesser as pp
from . import Visualizer as vis
from . import Merger as mrg
from . import DataExporter as de

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

    output_size = [config['int_output_x_size'], config['int_output_y_size'],3]  #Desired output size (width and height) is read from the config file

    path_rgb = config['str_dir_rgb'] # Directory where RGB images are stored
    path_thermal = config['str_dir_thermal'] # Directory where Thermal images are stored
    path_output = config['str_dir_output'] # Directory where results will be stored

    # Initialize class that loads all images
    data = dl.DataLoader(path_rgb,path_thermal,debug=False)
    # Preprocessor for RGB image with enhancing = False
    pp_c = pp.Preprocessor(output_size=output_size, enhancing=False)
    # Preprocessor for thermal image with enhancing specified by the user (through config file)
    pp_t = pp.Preprocessor(output_size=output_size, enhancing=config['bln_dofilter'])

    # Initialize Object Detection classes
    net_c = det.Detector('RGB',config['dbl_thermal_conf']) # RGB detector class
    net_t = det.Detector('Thermal',config['dbl_thermal_conf']) # Thermal detector class
    RGB_classNames = net_c.get_classes() # Yolo classnames are loaded through the RGB detector class

    # Initialize merger class for merging detections
    merge_c   = mrg.Merger( config['dbl_rgb_conf'],     config['dbl_rgb_nms']) # Merger for RGB image detections
    merge_t   = mrg.Merger( config['dbl_thermal_conf'], config['dbl_thermal_nms']) # Merger for Thermal image detections
    merge_all = mrg.Merger( 0.0,                        config['dbl_merge_nms']) # Merger for combining both RGB and Thermal image detections

    # Class for exporting the images & detections
    exporter = de.DataExporter(config, path_output, RGB_classNames)

    for file_name, file_ext, img_c, img_t in data:

        # Preprocessing
        img_c_out, img_c = pp_c.process(img_c) #R GB image getting preprocessed
        img_t_out, img_t = pp_t.process(img_t) # Thermal image getting preprocessed

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c.copy())) # RGB detections going through NMS function
        det_t = merge_t.NMS(net_t.detect(img_t.copy())) # Thermal detections going through NMS function
        det_m = det_c+det_t # RGB and Thermal detections are concatenated
        det_m = merge_all.NMS(det_m) # Concatenated detections going through NMS Function

        # Progress window
        V = vis.ProgressWindow(img_c, img_t, img_t_out, det_c, det_t, det_m, file_name+file_ext, RGB_classNames, data.progress, config)

        # Export data
        exporter.export(output_size,file_name, file_ext,det_m, img_t_out, img_t, config)

    print('Labeling completed.')
