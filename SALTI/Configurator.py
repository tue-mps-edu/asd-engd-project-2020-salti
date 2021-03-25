from configparser import ConfigParser
from tkinter import *
import os

def saveconfig(parser, config):
    for option in config:
        parser.set('Config',option,str(config[option].get()))
    cfgfile = open('config.ini','w')
    parser.write(cfgfile)
    cfgfile.close()

def ConfigSectionMap(parser, section):
    dict = {}
    options = parser.options(section)
    for option in options:
        type = option[0:3]
        if type == 'str':
            dict[option] = StringVar()
        elif type == 'dbl':
            dict[option] = DoubleVar()
        elif type == 'int':
            dict[option] = IntVar()
        elif type == 'bln':
            dict[option] = BooleanVar()
        else:
            raise NotImplementedError
        dict[option].set(parser.get(section, option))
    return dict


def ConfigSectionMapPythonvars(parser, section):
    dict = {}
    options = parser.options(section)
    for option in options:
        type = option[0:3]
        if type == 'str':
            dict[option] = str(parser.get(section, option))
        elif type == 'dbl':
            dict[option] = float(str(parser.get(section, option)))
        elif type == 'int':
            dict[option] = int(parser.get(section, option))
        elif type == 'bln':
            if parser.get(section, option) == "True":
                dict[option] = True
            else:
                dict[option] = False
            # dict[option] = bool(parser.get(section, option))
        else:
            raise NotImplementedError
    return dict

