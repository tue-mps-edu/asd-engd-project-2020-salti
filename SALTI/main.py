from GUI import *

def main():
    # Read the configuration file
    parser = configparser.ConfigParser()
    parser.read('config.ini')

    root = Tk()     # Initialize widget
    root.title("SALTI")
    dirs, thres, outputs = ConfigSectionMap(parser)

    create_gui(root,parser,dirs,thres,outputs)

    root.mainloop()

if __name__ == "__main__":
    main()
