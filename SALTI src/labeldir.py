import argparse

def labeldir(save_txt=False, save_img=False):
    '''
    This is the main function for the labeling program
    '''

    print("Input the main code here!")




if __name__ == '__main__':
    '''
    This can be used for running the function with different arguments.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    labeldir()
