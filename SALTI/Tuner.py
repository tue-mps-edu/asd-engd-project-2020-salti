import glob
from GUI import *
from Validator import *
import shutil
from Configurator import ConfigSectionMapPythonvars


# SALTI_path = os.path.join(os.getcwd(), os.path.dirname(__file__))
dir = os.chdir("..")
print(os.getcwd())
# os.chdir(SALTI_path)

# Define configuration file
config_file = 'config.ini'

# Read the configuration file
if os.path.isfile(config_file):
    parser = ConfigParser()
    parser.read(config_file)
else:
    print("ERROR: Config file not found at " + config_file + '!')
    raise FileExistsError

config = ConfigSectionMapPythonvars(parser, 'Config')

config['str_dir_rgb'] = "D:/Tuner2/test_images/color"
config['str_dir_thermal'] = "D:/Tuner2/test_images/thermal"
config['str_dir_output'] = "D:/Tuner2/test_images/Output"

#Setup the grid search variables
Steps=1
rgb_conf_list=np.linspace(0.05,0.25,Steps)
rgb_nms_list=np.linspace(0.05,0.5,Steps)
thermal_conf_list=np.linspace(0.05,0.25,Steps)
thermal_nms_list=np.linspace(0.05,0.5,Steps)
merge_nms_list=np.linspace(0.2,0.4,Steps)

#Output results
CL = ["RGB conf", "RGB NMS", "Thermal conf", "Thermal NMS", "Merge NMS", "TP", "FP", "FN" ,"Precision","Recall","Accuracy","F1"]
df = pd.DataFrame(columns=CL)

for rgb_conf in rgb_conf_list:
    for rgb_nms in rgb_nms_list:
        for thermal_conf in thermal_conf_list:
            for thermal_nms in thermal_nms_list:
                for merge_nms in merge_nms_list:

                    # Putting the current simulation's settings inside the configurator
                    config['dbl_rgb_conf'] = rgb_conf
                    config['dbl_rgb_nms'] = rgb_nms
                    config['dbl_thermal_conf'] = thermal_conf
                    config['dbl_thermal_nms'] = thermal_nms
                    config['dbl_merge_nms'] = merge_nms

                    #Run Salti
                    SALTI(config)

                    #Find the most recent subfolder inside the output folder
                    recent_subfolder = max(glob.glob(os.path.join(config['str_dir_output'], '*/')), key=os.path.getmtime)

                    # Copy/Paste the ground truth labels into the most recent subfolder
                    for root, dirs, files in os.walk(r"D:\Tuner2\test_images\labels", topdown=False):
                        for name in files:
                            source_file=os.path.join(root, name)
                            shutil.copy(source_file, recent_subfolder)

                    #Running the Validator on the most recent subfolder
                    TP,FP,FN,Precision,Recall,Accuracy,F1 = Validate(recent_subfolder,'.jpg',0.9)

                    df = df.append(pd.Series([rgb_conf,rgb_nms,thermal_conf,thermal_nms,merge_nms,TP,FP,FN,Precision,Recall,Accuracy,F1],
                                             index=df.columns), ignore_index=True)


#Saving the results as a csv file in the end
df.to_csv(os.path.join(config['str_dir_output'], 'Grid_Results.csv'))



