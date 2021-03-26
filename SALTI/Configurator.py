import tkinter as tk

def saveconfig(parser, config):
    '''
    Saves the parser configuration to the config.ini file

    Inputs:
    - (parser): configuration parser
    - (dict): config dictionary with Tkinter variables

    Outputs:
    - (config.ini): configuration file
    '''

    # Loop over the config dictionary and get TKinter variables
    for option in config:
        parser.set('Config',option,str(config[option].get()))
    cfgfile = open('config.ini','w')
    parser.write(cfgfile)
    cfgfile.close()

def ConfigSectionMap(parser, section):
    '''
    Generates a dictionary with TKinter variables from the config parser

    Inputs:
    - (parser): configuration parser object
    - (str): section name to generate map for

    Outputs:
    - (dict): dictionary with configuration
    '''
    dict = {}
    options = parser.options(section)
    for option in options:
        type = option[0:3]
        if type == 'str':
            dict[option] = tk.StringVar()
        elif type == 'dbl':
            dict[option] = tk.DoubleVar()
        elif type == 'int':
            dict[option] = tk.IntVar()
        elif type == 'bln':
            dict[option] = tk.BooleanVar()
        else:
            raise NotImplementedError
        dict[option].set(parser.get(section, option))
    return dict


def ConfigSectionMapPythonvars(parser, section):
    '''
    Generates a dictionary with default Python variables from the config parser

    Inputs:
    - (parser): configuration parser object
    - (str): section name to generate map for

    Outputs:
    - (dict): dictionary with configuration
    '''
    dict = {}
    options = parser.options(section)
    for option in options:
        type = option[0:3]
        if type == 'str':
            dict[option] = str(parser.get(section, option))
        elif type == 'dbl':
            dict[option] = float(str(parser.get(section, option)))
        elif type == 'int':
            dict[option] = int(parser.get(section, option))
        elif type == 'bln':
            if parser.get(section, option) == "True":
                dict[option] = True
            else:
                dict[option] = False
        else:
            raise NotImplementedError
    return dict

