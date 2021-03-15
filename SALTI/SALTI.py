from Detector import Detector
from DataLoader import DataLoader
from Preprocesser import Preprocessor
from Visualizer import ProgressWindow
from Merger import Merger
from DataExporter import DataExporter


def SALTI(dirs, thres, outputs):

    output_size = [outputs['x_size'].get(), outputs['y_size'].get()]

    path_rgb = dirs['rgb'].get()
    path_thermal = dirs['thermal'].get()
    path_output = dirs['output'].get()

    data = DataLoader(path_rgb,path_thermal,debug=True)

    do_resize = (data.img_size[0]==output_size[0] and data.img_size[1]==output_size[1])
    pp = Preprocessor(output_size=output_size, resize=do_resize)

    net_c = Detector('RGB')
    net_t = Detector('Thermal', thres['thermal_conf'].get(), thres['thermal_nms'].get())
    RGB_classNames = net_c.get_classes() #Should be improved and retrieved through a getter (make it private attribute)

    # Initialize merger
    merge_c   = Merger( thres['rgb_conf'].get(),     thres['rgb_nms'].get())
    merge_t   = Merger( thres['thermal_conf'].get(), thres['thermal_nms'].get())
    merge_all = Merger( 0.0,                         thres['merge_nms'].get())

    label_type = 'PascalVOC'
    exporter = DataExporter(label_type, path_output , RGB_classNames)


    for file_name, file_ext, img_c, img_t in data:
        #img_output = img_t.copy()
        if do_resize:
            img_c = pp.process(img_c)
            img_t = pp.process(img_t)

        # Do detections and apply non-maximum suppression on BBOXes
        det_c = merge_c.NMS(net_c.detect(img_c.copy()))
        det_t = net_t.detect(img_t.copy())
        det_m = det_c+det_t
        det_m = merge_all.NMS(det_m)
        
        # Progress window
        V = ProgressWindow(img_c, img_t, det_c, det_t, det_m, RGB_classNames, data.progress)

        # Export data
        exporter.export(img_t.shape,file_name, file_ext,det_m, img_t)



