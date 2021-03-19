from GUI import *
from multiprocessing import Process

def main():
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read('config.ini')

    root = Tk()     # Initialize widget
    root.title("SALTI")
    config = ConfigSectionMap(parser, 'Config')
    saveconfig(parser,config)
    create_gui(root,parser,config)

    root.mainloop()

if __name__ == "__main__":
    p_gui = Process(target=main())
    p_gui.start()
    #main()
