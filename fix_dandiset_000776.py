# In dandiset 000776, the data is invalid against the spec. See https://github.com/dandi/helpdesk/issues/126
# To fix the files in dandiset 000776, run this script on dandihub. This script performs the following data surgery
# steps:
#
# 1. Replace the "order_optical_channels" link from all `ImagingVolume` objects with a subgroup
#    that is the link target (at /processing/NeuroPAL/OpticalChannelRefs)
# 2. Add the "ndx-multichannel-volume" version 0.1.12 schema
# 3. Remove the "ndx-multichannel-volume" version 0.1.9 schema
# 4. Remove the "order_optical_channels" group from all "MultiChannelVolume" objects
# 5. Remove the "OpticalChannelRefs" group within "/processing"

import glob
import h5py
from pynwb import NWBHDF5IO, validate, get_type_map
from hdmf.backends.hdf5.h5_utils import H5SpecWriter
from hdmf.backends.utils import NamespaceToBuilderHelper
import argparse
from typing import Union

# download ndx-multichannel-volume extension 0.1.12 spec files here:
# https://github.com/focolab/ndx-multichannel-volume/tree/main/spec
type_map = get_type_map()
type_map.load_namespaces("/Users/danielysprague/foco_lab/ndx-multichannel-volume/spec/ndx-multichannel-volume.namespace.yaml")
ns_catalog = type_map.namespace_catalog


def replace_object(_: str, h5obj: Union[h5py.Dataset, h5py.Group]):
    if isinstance(h5obj, h5py.Group):
        if h5obj.attrs.get("neurodata_type") == "ImagingVolume":
            if "order_optical_channels" in h5obj:
                link_type = h5obj.get("order_optical_channels", getlink=True)
                if not isinstance(link_type, h5py.SoftLink):
                    return
                link_target = link_type.path
                assert link_target == "/processing/NeuroPAL/OpticalChannelRefs" or link_target == "/processing/ProcessedImage/OpticalChannelRefs" or link_target == "/processing/CalciumActivity/OpticalChannelRefs"

                del h5obj["order_optical_channels"]
                h5obj.create_group("order_optical_channels")

                order_optical_channels = h5obj.file[link_target]
                for attr in order_optical_channels.attrs:
                    h5obj["order_optical_channels"].attrs[attr] = order_optical_channels.attrs[attr]
                h5obj["order_optical_channels"].create_dataset(
                    name="channels",
                    data=order_optical_channels["channels"][:],
                    dtype=order_optical_channels["channels"].dtype,
                )
        elif h5obj.name == "/specifications/ndx-multichannel-volume":
            if "0.1.9" in h5obj:
                del h5obj["0.1.9"]
            if "0.1.12" in h5obj:
                del h5obj["0.1.12"]
            ns_builder = NamespaceToBuilderHelper.convert_namespace(ns_catalog, "ndx-multichannel-volume")
            ns_group = h5obj.create_group("0.1.12")
            writer = H5SpecWriter(ns_group)
            ns_builder.export("namespace", writer=writer)

def delete_object(_: str, h5obj: Union[h5py.Dataset, h5py.Group]):
    if isinstance(h5obj, h5py.Group):
        if h5obj.attrs.get("neurodata_type") == "MultiChannelVolume":
            if "order_optical_channels" in h5obj:
                del h5obj["order_optical_channels"]
        elif h5obj.attrs.get("neurodata_type") == "ProcessingModule":
            if "OpticalChannelRefs" in h5obj:
                del h5obj["OpticalChannelRefs"]


def adjust_file(filepath):
    print("-------------------------------------------------------------------")
    print("Adjusting NWB file:", filepath)
    with h5py.File(filepath, "a") as f:
        f.visititems(replace_object)
        f.visititems(delete_object)


def validate_nwb(filepath):
    print("-------------------------------------------------------------------")
    print("Validating NWB file:", filepath)
    with NWBHDF5IO(filepath, "r", load_namespaces=True) as io:
        errors = validate(io=io, namespace="ndx-multichannel-volume")
        if errors:
            print("Errors found:")
            print(errors)
        else:
            print("No errors found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="path to the NWB file or directory of NWB files to adjust"
    )
    args = parser.parse_args()
    path = args.path

    if path.endswith(".nwb"):
        filepaths = [path]
    else:
        filepaths = glob.glob(path + "/**/*.nwb", recursive=True)

    print("Adjusting these NWB files:", filepaths, sep="\n")
    for filepath in filepaths:
        adjust_file(filepath=filepath)
        validate_nwb(filepath=filepath)


if __name__ == "__main__":
    main()