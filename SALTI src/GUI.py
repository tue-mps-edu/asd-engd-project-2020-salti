from tkinter import *
from tkinter import filedialog
from configurator import *


def update_dir_rgb(dirs):
    dirs['rgb'].set(filedialog.askdirectory(initialdir=dirs['rgb'],title="Select RGB images directory"))

def update_dir_thermal(dirs):
    dirs['thermal'].set(filedialog.askdirectory(initialdir=dirs['thermal'], title="Select Thermal images directory"))
    #dirs['thermal'].set(str(dirs['thermal']))

def update_dir_output(dirs):
    dirs['output'].set(filedialog.askdirectory(initialdir=dirs['thermal'], title="Select output directory"))
    #var_O.set(str(dirs['output']))

# Grid configuration
rows = {"C":1,
        "T":2,
        "O":3}
cols = {"dirheader":0,
        "dirprint":1,
        "dirbutton":2}

def GUI_add_directory(root, dirs):
    Label(root,text="    Image directories", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=0,columnspan=1)
    # Setting the RGB directory
    Label(root,text="RGB image directory:", justify=LEFT, anchor="w").grid(sticky = W,row=rows['C'],column=cols['dirheader'])
    Label(root,textvariable=dirs['rgb'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=rows['C'],column=cols['dirprint'])
    Button(root,text="Set RGB path",command=update_dir_rgb,width=15).grid(row=rows['C'],column=cols['dirbutton'])
    # Setting the thermal directory
    Label(root,text="Thermal image directory:", justify=LEFT).grid(sticky = W,row=rows['T'],column=cols['dirheader'])
    Label(root,textvariable=dirs['thermal'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=rows['T'],column=cols['dirprint'])
    Button(root,text="Set thermal path",command=update_dir_thermal,width=15).grid(row=rows['T'],column=cols['dirbutton'])
    # Setting the output directory
    Label(root,text="Output directory:", justify=LEFT).grid(sticky = W,row=rows['O'],column=cols['dirheader'])
    Label(root,textvariable=dirs['output'], justify=LEFT, anchor="w",padx=15).grid(sticky = W,row=rows['O'],column=cols['dirprint'])
    Button(root,text="Set output path",command=update_dir_output,width=15).grid(row=rows['O'],column=cols['dirbutton'])

scale_rows= [x for x in range(6,11)]

def GUI_add_scales(root, thres, scale_rows):
    Label(root,text=" ").grid(row=4,columnspan=3)
    Label(root,text="    Algorithm settings", font='Helvetica 18 bold', justify=LEFT, anchor="w").grid(sticky=W,row=5,columnspan=1)

    # RGB sliders
    Label(root,text="RGB NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=scale_rows[0],column=0)
    Scale(root,variable=thres['rgb_nms'],resolution=0.05,to=1,width=20,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=scale_rows[0],column=1)
    Label(root,text="RGB Confidence level:", justify=LEFT, anchor="w").grid(sticky = W,row=scale_rows[1],column=0)
    Scale(root,variable=thres['rgb_conf'],resolution=0.05,to=1,width=20,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=scale_rows[1],column=1)
    # Thermal Sliders
    Label(root,text="Thermal NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=scale_rows[2],column=0)
    Scale(root,variable=thres['thermal_nms'],resolution=0.05,to=1,width=20,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=scale_rows[2],column=1)
    Label(root,text="Thermal Confidence level:", justify=LEFT, anchor="w",width=25).grid(sticky = W,row=scale_rows[3],column=0)
    Scale(root,variable=thres['thermal_conf'],resolution=0.05,to=1,width=20,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=scale_rows[3],column=1)
    # Merge sliders
    Label(root,text="Merge NMS level:", justify=LEFT, anchor="w").grid(sticky = W,row=scale_rows[4],column=0)
    Scale(root,variable=thres['merge_nms'],resolution=0.05,to=1,width=20,showvalue=True,orient=HORIZONTAL).grid(sticky = W,row=scale_rows[4],column=1)

def GUI_add_buttons(root, scale_rows, dirs, thres):
    # Saving the configuration
    Button(root,text="Save configuration",command=saveconfig,width=15).grid(row=scale_rows[3],column=cols['dirbutton'])
    # Running SALTI
    #Button(root,text="RUN SALTI",command=SALTI,width=15,bg='#00d60e').grid(row=scale_rows[4],column=cols['dirbutton'])

