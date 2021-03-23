import os, glob
from GUI import *
import numpy as np
from Validator import *
import shutil
from Configurator import ConfigSectionMapPythonvars


SALTI_path = os.path.join(os.getcwd(), os.path.dirname(__file__))
os.chdir(SALTI_path)

# Define configuration file
config_file = 'config_tuner.ini'

# Read the configuration file
if os.path.isfile(config_file):
    parser = ConfigParser()
    parser.read(config_file)
else:
    print("ERROR: Config file not found at " + config_file + '!')
    raise FileExistsError

config = ConfigSectionMapPythonvars(parser, 'Config')
#config = tkinterDict_to_pythonDict (config)

intervals=3
rgb_conf_list=np.linspace(0.1,1,intervals,endpoint=False)
rgb_nms_list=np.linspace(0.1,1,intervals,endpoint=False)
thermal_conf_list=np.linspace(0.1,1,intervals,endpoint=False)
thermal_nms_list=np.linspace(0.1,1,intervals,endpoint=False)
merge_nms_list=np.linspace(0.1,1,intervals,endpoint=False)

CL = ["RGB conf", "RGB NMS", "Thermal conf", "Thermal NMS", "Merge NMS", "Precision","Recall","Accuracy","F1"]
df = pd.DataFrame(columns=CL)

for rgb_conf in rgb_conf_list:
    for rgb_nms in rgb_nms_list:
        for thermal_conf in thermal_conf_list:
            for thermal_nms in thermal_nms_list:
                for merge_nms in merge_nms_list:


                    config['dbl_rgb_conf'] = rgb_conf
                    config['dbl_rgb_nms'] = rgb_nms
                    config['dbl_thermal_conf'] = thermal_conf
                    config['dbl_thermal_nms'] = thermal_nms
                    config['dbl_merge_nms'] = merge_nms

                    SALTI(config)

                    recent_subfolder = max(glob.glob(os.path.join(config['str_dir_output'], '*/')), key=os.path.getmtime)

                    for root, dirs, files in os.walk(r"D:\Tuner\KAIST_DAY\GT_KAIST_DAY", topdown=False):
                        for name in files:
                            source_file=os.path.join(root, name)
                            shutil.copy(source_file, recent_subfolder)

                    Precision,Recall,Accuracy,F1 = Validate(recent_subfolder,'.jpg',0.9)

                    df = df.append(pd.Series([rgb_conf,rgb_nms,thermal_conf,thermal_nms,merge_nms,Precision,Recall,Accuracy,F1],
                                             index=df.columns), ignore_index=True)
                    print(df)

df.to_csv(os.path.join(config['str_dir_output'], 'Grid_Results.csv'))



