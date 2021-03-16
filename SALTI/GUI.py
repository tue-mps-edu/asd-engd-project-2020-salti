from tkinter import *
from tkinter import filedialog
from Configurator import *
from functools import partial
from SALTI import SALTI
import os
from sys import platform

def update_dir_rgb(config):
    config['str_dir_rgb'].set(filedialog.askdirectory(initialdir=config['str_dir_rgb'],title="Select RGB images directory"))

def update_dir_thermal(config):
    config['str_dir_thermal'].set(filedialog.askdirectory(initialdir=config['str_dir_thermal'], title="Select Thermal images directory"))

def update_dir_output(config):
    config['str_dir_output'].set(filedialog.askdirectory(initialdir=config['str_dir_thermal'], title="Select output directory"))

def create_gui(root, parser, config):
    '''
        BOOLEANS TO BE ADDED TO GUI
    '''
    #raise NotImplementedError
    #config['bln_preprocessing'] = BooleanVar()
    #config['bln_validationcopy'] = BooleanVar()
    #config['bln_enhancevisibility'] = BooleanVar()
    #config['int_output_x_size'] = IntVar()
    #config['int_output_x_size'].set(640)
    #config['int_output_y_size'] = IntVar()
    #config['int_output_y_size'].set(512)
    '''
        PATH SETTINGS
    '''
    r_path = 1
    col_label_path = 0
    col_text_path = 1
    col_button_path = 3
    span = 2
    # Header
    Label(root,text="    Image directories", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_path,columnspan=4)
    # Setting the RGB directory
    Label(root,text="RGB image directory:", justify=LEFT, anchor="w").grid(sticky = W,row=r_path+1,column=col_label_path)
    Label(root,textvariable=config['str_dir_rgb'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+1,column=col_text_path, columnspan=span)
    Button(root,text="Set RGB path",command=partial(update_dir_rgb, config),width=15).grid(row=r_path+1,column=col_button_path)
    # Setting the thermal directory
    Label(root,text="Thermal image directory:", justify=LEFT).grid(sticky = W,row=r_path+2,column=col_label_path)
    Label(root,textvariable=config['str_dir_thermal'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+2,column=col_text_path, columnspan=span)
    Button(root,text="Set thermal path",command=partial(update_dir_thermal, config),width=15).grid(row=r_path+2,column=col_button_path)
    # Setting the output directory
    Label(root,text="Output directory:", justify=LEFT).grid(sticky = W,row=r_path+3,column=col_label_path)
    Label(root,textvariable=config['str_dir_output'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+3,column=col_text_path, columnspan=span)
    Button(root,text="Set output path",command=partial(update_dir_output, config),width=15).grid(row=r_path+3,column=col_button_path)
    '''
        ONE LINE SPACING
    '''
    r_empty = r_path+4
    Label(root,text=" ").grid(row=r_empty,columnspan=3)
    '''
        ALGORITHM SETTINGS
    '''
    r_alg = r_empty+1
    # Header
    scale_width = 15
    scale_length = 130
    Label(root,text="    Algorithm settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_alg+1,column=0,columnspan=2)
    # Preprocessing check button
    Label(root,text="Preprocessing enabled:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+2,column=0)
    Checkbutton(root, width = 15, variable = config['bln_preprocessing'], justify=LEFT, anchor="w").grid(sticky=W, row=r_alg+2, column=1)
    # RGB sliders
    Label(root,text="RGB NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+3,column=0)
    Scale(root,variable=config['dbl_rgb_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+3,column=1)
    Label(root,text="RGB Confidence level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+4,column=0)
    Scale(root,variable=config['dbl_rgb_conf'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+4,column=1)
    # Thermal Sliders
    Label(root,text="Thermal NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+5,column=0)
    Scale(root,variable=config['dbl_thermal_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+5,column=1)
    Label(root,text="Thermal Confidence level:", justify=LEFT, anchor="w",width=25).grid(sticky = W,row=r_alg+6,column=0)
    Scale(root,variable=config['dbl_thermal_conf'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+6,column=1)
    # Merge sliders
    Label(root,text="Merge NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+7,column=0)
    Scale(root,variable=config['dbl_merge_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+7,column=1)
    '''
        OUTPUT SETTINGS
    '''
    #Label(root,text=" ").grid(row=r_empty,columnspan=2)
    Label(root,text=" ").grid(row=r_alg+8,columnspan=3)
    r_out = r_alg+9
    # Header
    Label(root,text="    Output settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_out,column=0,columnspan=2)
    # Output size
    Label(root,text="Image size x:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+1,column=0)
    Entry(root, textvariable=config['int_output_x_size'], justify=LEFT,width=20).grid(sticky = W,row=r_out+1,column=1)
    Label(root,text="Image size y:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+2,column=0)
    Entry(root, textvariable=config['int_output_y_size'], justify=LEFT,width=20).grid(sticky = W,row=r_out+2,column=1)
    # Output format
    OptionList = [
        "YOLO",
        "PascalVOC",
    ]
    Label(root,text="Output label format:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+3,column=0)
    om1 = OptionMenu(root,  config['str_label'], *OptionList)
    om1.config(width=15)
    om1.grid(sticky=W, row=r_out+3, column=1)
    # Validation checkbox
    Label(root,text="Validation enabled:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+4,column=0)
    Checkbutton(root, width = 15, variable = config['bln_validationcopy'], justify=LEFT, anchor="w").grid(sticky=W, row=r_out+4, column=1)
    # Output enhanced image
    Label(root,text="Enhanced visibility:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+5,column=0)
    Checkbutton(root, width = 15, variable = config['bln_enhancevisibility'], justify=LEFT, anchor="w").grid(sticky=W, row=r_out+5, column=1)
    # Padding
    Label(root,text="  ", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+6,column=3)
    Label(root,text="  ", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+5,column=4)


    '''
        RUNNING SALTI
    '''
    Button(root,text="Open output folder",command= partial(open_folder,config),width=15,font='Helvetica 11').grid(row=r_out+4,column=col_button_path)
    Button(root,text="RUN SALTI",command= partial(save_and_run, parser, config),width=15,font='Helvetica 11 bold').grid(row=r_out+5,column=col_button_path)


def open_folder(config):
    # Only tested for Windows!
    try:
        os.system("explorer "+str(config['str_dir_output'].get()))
    except:
        raise NotImplementedError


def save_and_run(parser,config):
    saveconfig(parser, config)

    dirs_dict=py_dictionaries(config) #Changing tkinter dictionaries to normal python dictionaries
    thres_dict = py_dictionaries(config) #Changing tkinter dictionaries to normal python dictionaries
    outputs_dict = py_dictionaries(config) #Changing tkinter dictionaries to normal python dictionaries

    SALTI(dirs_dict, thres_dict, outputs_dict)

def py_dictionaries(tkinter_dict):
    py_dict = {}
    for option in tkinter_dict:
        py_dict[option]=tkinter_dict[option].get()
    return py_dict
