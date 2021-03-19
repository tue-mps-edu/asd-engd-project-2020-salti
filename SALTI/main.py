from GUI import *
from multiprocessing import Process, current_process

def main():
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read('config.ini')
    salti_processes = []

    root = Tk()     # Initialize widget
    root.title("SALTI")
    config = ConfigSectionMap(parser, 'Config')
    saveconfig(parser,config)
    create_gui(root,parser,config, salti_processes)
    root.mainloop()

if __name__ == "__main__":
    p_gui = Process(target=main)
    p_gui.start()
    print('GUI started at PID '+str(p_gui.pid))
