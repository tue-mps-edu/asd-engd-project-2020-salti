from tkinter import *
import configparser
from configurator import *
from GUI import *


def main():
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read('config.ini')

    root = Tk()     # Initialize widget
    root.title("SALTI")
    dirs, thres, outputs = ConfigSectionMap(parser)

    GUI_add_directory(root, dirs)
    GUI_add_scales(root, thres, scale_rows)
    GUI_add_buttons(root, parser, scale_rows, dirs, thres, outputs)

    root.mainloop()

if __name__ == "__main__":
    main()
