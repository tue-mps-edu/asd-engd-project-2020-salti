from tkinter import *
import configparser
from configurator import *
from functools import partial
from GUI import *
from SALTI import SALTI

def main():
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read('config.ini')

    root = Tk()     # Initialize widget
    root.title("SALTI")
    dirs, thres, outputs = ConfigSectionMap(parser)

    GUI_add_directory(root, dirs)
    GUI_add_scales(root, thres, scale_rows)
    GUI_add_buttons(root, scale_rows, dirs, thres)
    Button(root,text="RUN SALTI",command= partial(SALTI, dirs, thres, outputs),width=15,font='Helvetica 11 bold').grid(row=scale_rows[4],column=cols['dirbutton'])
    root.mainloop()

if __name__ == "__main__":
    main()
