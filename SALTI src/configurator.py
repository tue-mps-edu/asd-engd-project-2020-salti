import configparser
from tkinter import *

def saveconfig(parser, dirs, thres, outputs):
    parser.set('Directories','rgb',dirs['rgb'].get())
    parser.set('Directories', 'thermal', dirs['thermal'].get())
    parser.set('Directories','output',dirs['output'].get())
    parser.set('Thresholds','rgb_nms',str(thres['rgb_nms'].get()))
    parser.set('Thresholds', 'rgb_conf', str(thres['rgb_conf'].get()))
    parser.set('Thresholds', 'thermal_nms', str(thres['thermal_nms'].get()))
    parser.set('Thresholds', 'thermal_conf', str(thres['thermal_conf'].get()))
    parser.set('Thresholds', 'merge_nms', str(thres['merge_nms'].get()))
    parser.set('Outputs', 'x_size', str(outputs['x_size'].get()))
    parser.set('Outputs', 'y_size', str(outputs['y_size'].get()))
    parser.set('Outputs', 'label', str(outputs['label'].get()))
    cfgfile = open('config.ini','w')
    parser.write(cfgfile)
    cfgfile.close()

def ConfigSectionMap(parser):
    dirs, thres, outputs = {},{},{}
    section = 'Directories'
    options = parser.options(section)
    for option in options:
        dirs[option] = StringVar()
        dirs[option].set(parser.get(section, option))
    section = 'Thresholds'
    options = parser.options(section)
    for option in options:
        thres[option] = DoubleVar()
        thres[option].set(parser.get(section, option))
    section = 'Outputs'
    options = parser.options(section)
    for option in options:
        outputs[option] = StringVar()
        outputs[option].set(parser.get(section, option))
    return dirs, thres, outputs
