def UserValidation(variables):
    import pandas as pd
    import os
    #this is code for reading output files from YOlO and from the GUI
    #YOLO

    #GUI


    a = pd.read_csv("GOPR0376_frame_000001_rgb_anon.txt", delim_whitespace=True,
                    names=["classes", "x_c", "ÿ_c", "w", "h"])
    a['filename'] = os.path.basename("GOPR0376_frame_000001_rgb_anon.txt")
    # pd.DataFrame("GOPR0376_frame_000001_rgb_anon")

    print(a)