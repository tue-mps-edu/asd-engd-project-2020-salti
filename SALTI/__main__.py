import tkinter as tk
from . import GUI
from . import Configurator
from configparser import ConfigParser
import os

def main():
    '''
    Main function of the SALTI package.
    It initiates the configuration parser and sets up the GUI.
    '''
    # Change path to SALTI
    SALTI_path = os.path.join(os.getcwd(),os.path.dirname(__file__))
    os.chdir(SALTI_path)

    # Read the configuration file
    config_file = 'config.ini'
    if os.path.isfile(config_file):
        parser = ConfigParser()
        parser.read(config_file)
    else:
        print("ERROR: Config file not found at "+config_file+'!')
        raise FileExistsError

    # This will log al parallel running SALTI processes
    salti_processes = []

    # Initialize the GUI
    root = tk.Tk()     # Initialize widget
    root.title("SALTI")
    # Initialize configuration dict as TKinter variables
    config = Configurator.ConfigSectionMap(parser, 'Config')
    # Fill the GUI & run
    GUI.create_gui(root,parser,config, salti_processes)
    root.mainloop()

if __name__ == "__main__":
    main()
