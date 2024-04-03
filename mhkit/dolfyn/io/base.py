import numpy as np
import xarray as xr
import json
import os
import warnings


def _abspath(fname):
    return os.path.abspath(os.path.expanduser(fname))


def _get_filetype(fname):
    """
    Detects whether the file is a Nortek, Signature (Nortek), or RDI
    file by reading the first few bytes of the file.

    Returns
    =======
       None - Doesn't match any known pattern
       'signature' - for Nortek signature files
       'nortek' - for Nortek (Vec, AWAC) files
       'RDI' - for RDI files
       '<GIT-LFS pointer> - if the file looks like a GIT-LFS pointer.
    """

    with open(fname, "rb") as rdr:
        bytes = rdr.read(40)
    code = bytes[:2].hex()
    if code in ["7f79", "7f7f"]:
        return "RDI"
    elif code in ["a50a"]:
        return "signature"
    elif code in ["a505"]:
        # AWAC
        return "nortek"
    elif bytes == b"version https://git-lfs.github.com/spec/":
        return "<GIT-LFS pointer>"
    else:
        return None


def _find_userdata(filename, userdata=True):
    # This function finds the file to read
    if userdata:
        for basefile in [filename.rsplit(".", 1)[0], filename]:
            jsonfile = basefile + ".userdata.json"
            if os.path.isfile(jsonfile):
                return _read_userdata(jsonfile)

    elif isinstance(userdata, (str,)) or hasattr(userdata, "read"):
        return _read_userdata(userdata)
    return {}


def _read_userdata(fname):
    """
    Reads a userdata.json file and returns the data it contains as a
    dictionary.
    """
    with open(fname) as data_file:
        data = json.load(data_file)
    for nm in ["body2head_rotmat", "body2head_vec"]:
        if nm in data:
            new_name = "inst" + nm[4:]
            warnings.warn(
                f"{nm} has been deprecated, please change this to {new_name} \
                    in {fname}."
            )
            data[new_name] = data.pop(nm)
    if "inst2head_rotmat" in data:
        if data["inst2head_rotmat"] in ["identity", "eye", 1, 1.0]:
            data["inst2head_rotmat"] = np.eye(3)
        else:
            data["inst2head_rotmat"] = np.array(data["inst2head_rotmat"])
    if "inst2head_vec" in data and type(data["inst2head_vec"]) != list:
        data["inst2head_vec"] = list(data["inst2head_vec"])

    return data


def _handle_nan(data):
    """
    Finds trailing nan's that cause issues in running the rotation
    algorithms and deletes them.
    """
    nan = np.zeros(data["coords"]["time"].shape, dtype=bool)
    l = data["coords"]["time"].size

    if any(np.isnan(data["coords"]["time"])):
        nan += np.isnan(data["coords"]["time"])

    # Required for motion-correction algorithm
    var = ["accel", "angrt", "mag"]
    for key in data["data_vars"]:
        if any(val in key for val in var):
            shp = data["data_vars"][key].shape
            if shp[-1] == l:
                if len(shp) == 1:
                    if any(np.isnan(data["data_vars"][key])):
                        nan += np.isnan(data["data_vars"][key])
                elif len(shp) == 2:
                    if any(np.isnan(data["data_vars"][key][-1])):
                        nan += np.isnan(data["data_vars"][key][-1])
    trailing = np.cumsum(nan)[-1]

    if trailing > 0:
        data["coords"]["time"] = data["coords"]["time"][:-trailing]
        for key in data["data_vars"]:
            if data["data_vars"][key].shape[-1] == l:
                data["data_vars"][key] = data["data_vars"][key][..., :-trailing]

    return data


def _create_dataset(data):
    """
    Creates an xarray dataset from dictionary created from binary
    readers.
    Direction 'dir' coordinates are set in `set_coords`
    """

    tag = ["_avg", "_b5", "_echo", "_bt", "_gps", "_altraw", "_altraw_avg", "_sl"]

    ds_dict = {}
    for key in data["coords"]:
        ds_dict[key] = {"dims": (key), "data": data["coords"][key]}

    # Set various coordinate frames
    if "n_beams_avg" in data["attrs"]:
        beams = data["attrs"]["n_beams_avg"]
    else:
        beams = data["attrs"]["n_beams"]
    n_beams = max(min(beams, 4), 3)
    beams = np.arange(1, n_beams + 1, dtype=np.int32)

    ds_dict["beam"] = {"dims": ("beam"), "data": beams}
    ds_dict["dir"] = {"dims": ("dir"), "data": beams}
    data["units"].update({"beam": "1", "dir": "1"})
    data["long_name"].update({"beam": "Beam Reference Frame", "dir": "Reference Frame"})

    # Iterate through data variables and add them to new dictionary
    for key in data["data_vars"]:
        # orientation matrices
        if "mat" in key:
            if "inst" in key:  # beam2inst & inst2head orientation matrices
                if "x1" not in ds_dict:
                    ds_dict["x1"] = {"dims": ("x1"), "data": beams}
                    ds_dict["x2"] = {"dims": ("x2"), "data": beams}

                ds_dict[key] = {"dims": ("x1", "x2"), "data": data["data_vars"][key]}
                data["units"].update({key: "1"})
                data["long_name"].update({key: "Rotation Matrix"})

            elif "orientmat" in key:  # earth2inst orientation matrix
                if any(val in key for val in tag):
                    tg = "_" + key.rsplit("_")[-1]
                else:
                    tg = ""

                ds_dict["earth"] = {"dims": ("earth"), "data": ["E", "N", "U"]}
                ds_dict["inst"] = {"dims": ("inst"), "data": ["X", "Y", "Z"]}
                ds_dict[key] = {
                    "dims": ("earth", "inst", "time" + tg),
                    "data": data["data_vars"][key],
                }
                data["units"].update(
                    {"earth": "1", "inst": "1", key: data["units"]["orientmat"]}
                )
                data["long_name"].update(
                    {
                        "earth": "Earth Reference Frame",
                        "inst": "Instrument Reference Frame",
                        key: data["long_name"]["orientmat"],
                    }
                )

        # quaternion units never change
        elif "quaternions" in key:
            if any(val in key for val in tag):
                tg = "_" + key.rsplit("_")[-1]
            else:
                tg = ""

            if "q" not in ds_dict:
                ds_dict["q"] = {"dims": ("q"), "data": ["w", "x", "y", "z"]}
                data["units"].update({"q": "1"})
                data["long_name"].update({"q": "Quaternion Vector Components"})

            ds_dict[key] = {"dims": ("q", "time" + tg), "data": data["data_vars"][key]}
            data["units"].update({key: data["units"]["quaternions"]})
            data["long_name"].update({key: data["long_name"]["quaternions"]})

        else:
            shp = data["data_vars"][key].shape
            if len(shp) == 1:  # 1D variables
                if "_altraw_avg" in key:
                    tg = "_altraw_avg"
                elif any(val in key for val in tag):
                    tg = "_" + key.rsplit("_")[-1]
                else:
                    tg = ""
                ds_dict[key] = {"dims": ("time" + tg), "data": data["data_vars"][key]}

            elif len(shp) == 2:  # 2D variables
                if key == "echo":
                    ds_dict[key] = {
                        "dims": ("range_echo", "time_echo"),
                        "data": data["data_vars"][key],
                    }
                elif key == "samp_altraw":
                    ds_dict[key] = {
                        "dims": ("n_altraw", "time_altraw"),
                        "data": data["data_vars"][key],
                    }
                elif key == "samp_altraw_avg":
                    ds_dict[key] = {
                        "dims": ("n_altraw_avg", "time_altraw_avg"),
                        "data": data["data_vars"][key],
                    }

                # ADV/ADCP instrument vector data, bottom tracking
                elif shp[0] == n_beams and not any(val in key for val in tag[:3]):
                    if "bt" in key and "time_bt" in data["coords"]:
                        tg = "_bt"
                    else:
                        tg = ""
                    if any(
                        key.rsplit("_")[0] in s
                        for s in ["amp", "corr", "dist", "prcnt_gd"]
                    ):
                        dim0 = "beam"
                    else:
                        dim0 = "dir"
                    ds_dict[key] = {
                        "dims": (dim0, "time" + tg),
                        "data": data["data_vars"][key],
                    }

                # ADCP IMU data
                elif shp[0] == 3:
                    if not any(val in key for val in tag):
                        tg = ""
                    else:
                        tg = [val for val in tag if val in key]
                        tg = tg[0]

                    if "dirIMU" not in ds_dict:
                        ds_dict["dirIMU"] = {"dims": ("dirIMU"), "data": [1, 2, 3]}
                        data["units"].update({"dirIMU": "1"})
                        data["long_name"].update({"dirIMU": "Reference Frame"})

                    ds_dict[key] = {
                        "dims": ("dirIMU", "time" + tg),
                        "data": data["data_vars"][key],
                    }

                elif "b5" in tg:
                    ds_dict[key] = {
                        "dims": ("range_b5", "time_b5"),
                        "data": data["data_vars"][key],
                    }

            elif len(shp) == 3:  # 3D variables
                if "vel" in key:
                    dim0 = "dir"
                else:  # amp, corr, prcnt_gd, status
                    dim0 = "beam"

                if not any(val in key for val in tag) or ("_avg" in key):
                    if "_avg" in key:
                        tg = "_avg"
                    else:
                        tg = ""
                    ds_dict[key] = {
                        "dims": (dim0, "range" + tg, "time" + tg),
                        "data": data["data_vars"][key],
                    }

                elif "b5" in key:
                    # "vel_b5" sometimes stored as (1, range_b5, time_b5)
                    ds_dict[key] = {
                        "dims": ("range_b5", "time_b5"),
                        "data": data["data_vars"][key][0],
                    }
                elif "sl" in key:
                    ds_dict[key] = {
                        "dims": (dim0, "range_sl", "time"),
                        "data": data["data_vars"][key],
                    }
                else:
                    warnings.warn(f"Variable not included in dataset: {key}")

    # Create dataset
    ds = xr.Dataset.from_dict(ds_dict)

    # Assign data array attributes
    for key in ds.variables:
        for md in ["units", "long_name", "standard_name"]:
            if key in data[md]:
                ds[key].attrs[md] = data[md][key]
            if len(ds[key].shape) > 1:
                ds[key].attrs["coverage_content_type"] = "physicalMeasurement"
            try:  # make sure ones with tags get units
                tg = "_" + key.rsplit("_")[-1]
                if any(val in key for val in tag):
                    ds[key].attrs[md] = data[md][key[: -len(tg)]]
            except:
                pass

    # Assign coordinate attributes
    for ky in ds.dims:
        ds[ky].attrs["coverage_content_type"] = "coordinate"
    r_list = [r for r in ds.coords if "range" in r]
    for ky in r_list:
        ds[ky].attrs["units"] = "m"
        ds[ky].attrs["long_name"] = "Profile Range"
        ds[ky].attrs["description"] = "Distance to the center of each depth bin"
    time_list = [t for t in ds.coords if "time" in t]
    for ky in time_list:
        ds[ky].attrs["units"] = "seconds since 1970-01-01 00:00:00"
        ds[ky].attrs["long_name"] = "Time"
        ds[ky].attrs["standard_name"] = "time"

    # Set dataset metadata
    ds.attrs = data["attrs"]

    return ds
