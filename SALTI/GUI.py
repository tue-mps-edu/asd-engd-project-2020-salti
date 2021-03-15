from tkinter import *
from tkinter import filedialog
from Configurator import *
from functools import partial
from SALTI import SALTI
import os
from sys import platform

def update_dir_rgb(dirs):
    dirs['rgb'].set(filedialog.askdirectory(initialdir=dirs['rgb'],title="Select RGB images directory"))

def update_dir_thermal(dirs):
    dirs['thermal'].set(filedialog.askdirectory(initialdir=dirs['thermal'], title="Select Thermal images directory"))

def update_dir_output(dirs):
    dirs['output'].set(filedialog.askdirectory(initialdir=dirs['thermal'], title="Select output directory"))

def create_gui(root, parser, dirs, thres, outputs):
    '''
        BOOLEANS TO BE ADDED TO GUI
    '''
    #raise NotImplementedError
    bool_pp = BooleanVar()
    bool_val = BooleanVar()
    bool_enh = BooleanVar()
    text_var_x = IntVar()
    text_var_x.set(640)
    text_var_y = IntVar()
    text_var_y.set(512)
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
    Label(root,textvariable=dirs['rgb'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+1,column=col_text_path, columnspan=span)
    Button(root,text="Set RGB path",command=partial(update_dir_rgb, dirs),width=15).grid(row=r_path+1,column=col_button_path)
    # Setting the thermal directory
    Label(root,text="Thermal image directory:", justify=LEFT).grid(sticky = W,row=r_path+2,column=col_label_path)
    Label(root,textvariable=dirs['thermal'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+2,column=col_text_path, columnspan=span)
    Button(root,text="Set thermal path",command=partial(update_dir_thermal, dirs),width=15).grid(row=r_path+2,column=col_button_path)
    # Setting the output directory
    Label(root,text="Output directory:", justify=LEFT).grid(sticky = W,row=r_path+3,column=col_label_path)
    Label(root,textvariable=dirs['output'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=r_path+3,column=col_text_path, columnspan=span)
    Button(root,text="Set output path",command=partial(update_dir_output, dirs),width=15).grid(row=r_path+3,column=col_button_path)
    '''
        ONE LINE SPACING
    '''
    r_empty = r_path+4
    Label(root,text=" ").grid(row=r_empty,columnspan=4)
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
    Checkbutton(root, width = 15, variable = bool_pp, justify=LEFT, anchor="w").grid(sticky=W, row=r_alg+2, column=1)
    # RGB sliders
    Label(root,text="RGB NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+3,column=0)
    Scale(root,variable=thres['rgb_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+3,column=1)
    Label(root,text="RGB Confidence level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+4,column=0)
    Scale(root,variable=thres['rgb_conf'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+4,column=1)
    # Thermal Sliders
    Label(root,text="Thermal NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+5,column=0)
    Scale(root,variable=thres['thermal_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+5,column=1)
    Label(root,text="Thermal Confidence level:", justify=LEFT, anchor="w",width=25).grid(sticky = W,row=r_alg+6,column=0)
    Scale(root,variable=thres['thermal_conf'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+6,column=1)
    # Merge sliders
    Label(root,text="Merge NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+7,column=0)
    Scale(root,variable=thres['merge_nms'],resolution=0.05,to=1,width=scale_width, length=scale_length,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=r_alg+7,column=1)
    '''
        OUTPUT SETTINGS
    '''
    #Label(root,text=" ").grid(row=r_empty,columnspan=2)
    r_out = r_alg+8
    # Header
    Label(root,text="    Output settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_out,column=0,columnspan=2)
    # Output size
    Label(root,text="Image size x:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+1,column=0)
    Entry(root, textvariable=text_var_x, justify=LEFT,width=20).grid(sticky = W,row=r_out+1,column=1)
    Label(root,text="Image size y:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+2,column=0)
    Entry(root, textvariable=text_var_y, justify=LEFT,width=20).grid(sticky = W,row=r_out+2,column=1)
    # Output format
    OptionList = [
        "YOLO",
        "PascalVOC",
    ]
    Label(root,text="Output label format:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+3,column=0)
    om1 = OptionMenu(root,  outputs['label'], *OptionList)
    om1.config(width=15)
    om1.grid(sticky=W, row=r_out+3, column=1)
    # Validation checkbox
    Label(root,text="Validation enabled:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+4,column=0)
    Checkbutton(root, width = 15, variable = bool_val, justify=LEFT, anchor="w").grid(sticky=W, row=r_out+4, column=1)
    # Output enhanced image
    Label(root,text="Enhanced visibility:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+5,column=0)
    Checkbutton(root, width = 15, variable = bool_enh, justify=LEFT, anchor="w").grid(sticky=W, row=r_out+5, column=1)

    '''
        RUNNING SALTI
    '''
    Button(root,text="Open output folder",command= partial(open_folder,dirs),width=15,font='Helvetica 11').grid(row=r_out+4,column=col_button_path)
    Button(root,text="RUN SALTI",command= partial(save_and_run, parser, dirs, thres, outputs),width=15,font='Helvetica 11 bold').grid(row=r_out+5,column=col_button_path)

def open_folder(dirs):
    # Only tested for Windows!
    try:
        os.system("explorer "+str(dirs['output'].get()))
    except:
        raise NotImplementedError


def save_and_run(parser,dirs,thres,outputs):
    saveconfig(parser,dirs, thres, outputs)

    dirs_dict=py_dictionaries(dirs) #Changing tkinter dictionaries to normal python dictionaries
    thres_dict = py_dictionaries(thres) #Changing tkinter dictionaries to normal python dictionaries
    outputs_dict = py_dictionaries(outputs) #Changing tkinter dictionaries to normal python dictionaries

    SALTI(dirs_dict, thres_dict, outputs_dict)

def py_dictionaries(tkinter_dict):
    py_dict = {}
    for option in tkinter_dict:
        py_dict[option]=tkinter_dict[option].get()
    return py_dict
