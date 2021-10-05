import mrcfile
import zarr
import numpy as np



def mrc_to_zarr(input_mrc_list, output_zarr, input_label_list=None, labels_upside_down = True, flipy=None):
    """
    Convert a group of mrc files (data and labels) to a zarr file.

    Assume file names are of the form: '<identifier>_<datatype>.mrc'

    input_mrc_list - list (or singular string) of mrc files to convert
    input_label_list - list (or singular string) of mrc files to convert. Labels can be for a subset of data files.
    """
    if flipy==None:
        flipy=[]
    root = zarr.open_group(output_zarr, mode="w")
    root.create_group("data")
    root.create_group("labels")
    if type(input_mrc_list) is str:
        input_mrc_list = [input_mrc_list]
    if type(input_label_list) is str:
        input_label_list = [input_label_list]
    for index,input_mrc in enumerate(input_mrc_list):
        print(input_mrc)
        with mrcfile.open(input_mrc, mode="r", permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
            if index in flipy:
                data = np.flip(data, axis=1)
                print(f"Flipping y axis for {index},{input_mrc}")
            data = np.flip(data, axis=0)
            dataname = input_mrc.split("/")[-1].split("_")[0]
            root["data"].create_dataset(dataname, data=data, chunks=(1,)+(data.shape[1:]))
    if type(input_label_list) is list:
        for input_label in input_label_list:
            with mrcfile.open(input_label, mode="r", permissive=True) as mrc:
                dataname = input_label.split("/")[-1].split("_")[0]
                labels = mrc.data
                if labels_upside_down:
                    labels = np.flip(labels, axis=0)
                root["labels"].create_dataset(dataname, data=labels, chunks=(1,)+(labels.shape[1:]))

    print(root.tree())
    return root
    

