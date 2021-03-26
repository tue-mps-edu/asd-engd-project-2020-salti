import tkinter
from tkinter import filedialog
from Configurator import *
from functools import partial
from SALTI import SALTI
import os
from multiprocessing import Process


def update_dir_rgb(config):
    '''Update TKinter variable that contains the RGB image directory'''
    config['str_dir_rgb'].set(filedialog.askdirectory(initialdir=config['str_dir_rgb'],title="Select RGB images directory"))

def update_dir_thermal(config):
    '''Update TKinter variable that contains the Thermal image directory'''
    config['str_dir_thermal'].set(filedialog.askdirectory(initialdir=config['str_dir_thermal'], title="Select Thermal images directory"))

def update_dir_output(config):
    '''Update TKinter variable that contains the output directory'''
    config['str_dir_output'].set(filedialog.askdirectory(initialdir=config['str_dir_thermal'], title="Select output directory"))

def create_gui(root, parser, config, salti_processes):
    '''
    Add all GUI components to the root window
    Arguments:
        (tkinter window) Root window
        (parser) Configuration parser object
        (dict) configuration file with TKinter variables
        (list) log of active SALTI processes
    '''


    # Variables for grid alignment
    ''' Adding buttons for paths '''
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

    # Add spacing between sections
    r_empty = r_path+4
    Label(root,text=" ").grid(row=r_empty,columnspan=3)

    ''' ALGORITHM SETTINGS SECTION '''
    r_alg = r_empty+1
    # Header
    scale_width = 15
    scale_length = 130
    Label(root,text="    Algorithm settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_alg+1,column=0,columnspan=2)
    # Preprocessing check button
    Label(root,text="Filter thermal image:", justify=LEFT, anchor="w").grid(sticky = W,row=r_alg+2,column=0)
    Checkbutton(root, width = 15, variable = config['bln_dofilter'], justify=LEFT, anchor="w").grid(sticky=W, row=r_alg+2, column=1)
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

    ''' OUTPUT SETTINGS '''
    Label(root,text=" ").grid(row=r_alg+8,columnspan=3)
    r_out = r_alg+9
    # Header
    Label(root,text="    Output settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=r_out,column=0,columnspan=2)
    # Output size
    Label(root,text="Image width:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+1,column=0)
    Entry(root, textvariable=config['int_output_x_size'], justify=LEFT,width=20).grid(sticky = W,row=r_out+1,column=1)
    Label(root,text="Image height:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+2,column=0)
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
    Label(root,text="Create validation labels:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+4,column=0)
    Checkbutton(root, width = 15, variable = config['bln_validationcopy'], justify=LEFT, anchor="w").grid(sticky=W, row=r_out+4, column=1)
    # Output enhanced image
    Label(root,text="Save filtered images:", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+5,column=0)
    Checkbutton(root, width = 15, variable = config['bln_savefiltered'], justify=LEFT, anchor="w").grid(sticky=W, row=r_out+5, column=1)
    # Padding
    Label(root,text="  ", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+6,column=3)
    Label(root,text="  ", justify=LEFT, anchor="w").grid(sticky = W,row=r_out+5,column=4)

    ''' ADD RUNNING SALTI BUTTONS '''
    Button(root,text="Open output folder",command= partial(open_folder,config),width=15,font='Helvetica 11').grid(row=r_out+3,column=col_button_path)
    Button(root,text="RUN SALTI",command= partial(save_and_run, parser, config, salti_processes),width=15,font='Helvetica 11 bold').grid(row=r_out+4,column=col_button_path)
    Button(root,text="Stop SALTI",command=partial(stop_salti,salti_processes),width=15,font='Helvetica 11 bold').grid(row=r_out+5,column=col_button_path)



def open_folder(config):
    '''Function for opening a folder when clicking button in GUI'''
    try:
        path = str(config['str_dir_output'].get())
        path_w = path.replace(r"/", "\\")
        os.system("explorer "+path_w)
    except:
        raise NotImplementedError

def save_and_run(parser,config, salti_processes):
    '''Function that is called when user presses RUN SALTI'''

    # Save the latest Tkinter variables in the configuration file
    saveconfig(parser, config)
    # convert the Tkinter variables to default python
    config_dict = tkinterDict_to_pythonDict(config)
    # Create a new process for SALTI
    p_new = Process(target=SALTI, args=(config_dict,))
    # Log the process in list
    salti_processes.append(p_new)
    p_new.start()
    print('SALTI started at PID '+str(p_new.pid))

def stop_salti(salti_processes):
    '''Function that is called when user presses STOP SALTI'''
    for process in salti_processes:
        print('Killed SALTI at PID '+str(process.pid))
        process.kill()

def tkinterDict_to_pythonDict(tkinter_dict):
    '''Converting the configuration dictionary from TKinter to Python variables'''
    py_dict = {}
    for option in tkinter_dict:
        py_dict[option]=tkinter_dict[option].get()
    return py_dict

