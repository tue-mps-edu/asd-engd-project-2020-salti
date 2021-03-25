from GUI import *
import os

def main():
    # Change directory to SALTI
    SALTI_path = os.path.dirname(__file__)
    os.chdir(SALTI_path)

    # Define configuration file
    config_file = 'config.ini'

    # Read the configuration file
    if os.path.isfile(config_file):
        parser = ConfigParser()
        parser.read(config_file)
    else:
        print("ERROR: Config file not found at "+config_file+'!')
        raise FileExistsError

    salti_processes = []

    root = Tk()     # Initialize widget
    root.title("SALTI")
    config = ConfigSectionMap(parser, 'Config')
    saveconfig(parser,config)
    create_gui(root,parser,config, salti_processes)
    root.mainloop()

if __name__ == "__main__":
    main()
