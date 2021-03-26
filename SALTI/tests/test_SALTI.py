import SALTI
import cv2

img_c = cv2.imread(r'..\..\test_images\color\day_I00000.jpg')
img_t = cv2.imread(r'..\..\test_images\thermal\day_I00000.jpg')

def test_detections():
    a=SALTI.Detections([1,2],[3,4],[5,6])
    b=SALTI.Detections([11,122],[13,14],[15,16])
    L_before = len(a.confidences)
    c=a.append(b)
    assert len(a.confidences)==L_before, "Wrong length of detections"

def test_merger():
    b=SALTI.Detections([[1,2,3,4],[1,2,3,4]],[1,1],[0.1,0.7])
    a=SALTI.Merger(0.5,0.5)
    c=a.NMS(b)
    print(c.boxes)
    print(c.classes)
    print(c.confidences)
    print('length'+str(len(c.classes)))
    assert(len(c.classes)==1)

def test_preprocessor():
    output_size = [312, 312]
    img = cv2.imread('D:\KAIST\set00\V000\lwir\I00000.jpg')
    cv2.imshow("TEST", img)
    cv2.waitKey(100)
    PP = SALTI.Preprocessor(output_size, enhancing=True)
    img_pp, img_pp2 = PP.process(img)
    sz = img_pp.shape
    assert(sz[0]==output_size[0] and sz[1]==output_size[1])

def test_visualizer():
    classnames = ['car']
    det = SALTI.Detections([[200,200,300,300]],[0],[0.9])
    config = dict()
    config['bln_dofilter'] = True
    vis = SALTI.ProgressWindow(img_c, img_t, img_t, det, det, det, 'test.jpg',classnames,0.6,config)

    cv2.waitKey(10)
    print('pause')
    assert True, "Visualizer failed"

def test_thermal():
    net_t = SALTI.Detector('Thermal',0.5) # Thermal detector class
    det = net_t.detect(img_t)
    assert len(det.confidences)>0, "No detections"

def test_rgb():
    net_c = SALTI.Detector('RGB',0.5) # RGB detector class
    det = net_c.detect(img_t)
    assert len(det.confidences)>0, "No detections"

def test_dataloader():
    path_t = r'..\..\test_images\color'
    path_c = r'..\..\test_images\thermal'
    data = SALTI.DataLoader(path_c, path_t, debug=False)
    for file_name, file_ext, img_c, img_t in data:
        cv2.imshow("rgb",img_c)
        cv2.imshow("thermal",img_t)
        cv2.waitKey(100)
    assert True, "Failed test"

def test_exporter():
    classnames = ['car']
    det = SALTI.Detections([[200, 200, 300, 300]], [0], [0.9])
    out_path = r'..\..\test_images\color\outputs'
    filename = r'I00000'
    #config, output_path, classnames):
    config = dict()
    config['bln_validationcopy'] = True
    config['bln_savefiltered'] = True
    config['bln_dofilter'] = True
    config['str_label'] = 'YOLO'
    output_size = [640,512,3]
    E1 = SALTI.DataExporter(config, out_path, classnames)
    E1.export(output_size, filename, '.jpg', det, img_c, img_t, config)
    config['str_label'] = 'PascalVOC'
    E2 = SALTI.DataExporter(config, out_path, classnames)
    E2.export(output_size, filename, '.jpg', det, img_c, img_t, config)
    assert True, "Failed exporting"
