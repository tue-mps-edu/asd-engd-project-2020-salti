from GUI import *
from multiprocessing import Process, current_process
import SALTI

def main():
    # Define configuration file
    config_file = 'config.ini'
    SALTI_path = os.path.dirname(__file__)
    config_path = os.path.join(SALTI_path,config_file)
    # Read the configuration file
    if os.path.isfile(config_path):
        parser = ConfigParser()
        parser.read(config_path)
    else:
        print("Config file not found at "+config_file)

    salti_processes = []

    root = Tk()     # Initialize widget
    root.title("SALTI")
    config = ConfigSectionMap(parser, 'Config')
    saveconfig(parser,config)
    create_gui(root,parser,config, salti_processes)
    root.mainloop()

if __name__ == "__main__":
    main()
