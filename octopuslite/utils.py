import enum
import os
import re
from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter

import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import pandas as pd

@enum.unique
class Channels(enum.Enum):
    CH0 = 0
    CH1 = 1
    CH2 = 2
    CH3 = 3
    CH4 = 4
    CH5 = 5
    CH6 = 6
    CH7 = 7
    MASK7 = 93
    MASK6 = 94
    MASK5 = 95
    MASK4 = 96
    MASK3 = 97
    MASK2 = 98
    MASK1 = 99
    MASK0 = 100

def remove_outliers(x: np.ndarray) -> np.ndarray:
    """Remove bright outlier pixels from an image.

    Parameters
    ----------
    x : np.ndarray
        An input image containing bright outlier pixels.

    Returns
    -------
    x : np.ndarray
        An image with the bright outlier pixels removed.
    """
    med_x = median_filter(x, size=2)
    mask = x > med_x
    x = x * (1 - mask) + (mask * med_x)
    return x


def remove_background(x: np.ndarray) -> np.ndarray:
    """Remove background using a polynomial surface.

    Parameters
    ----------
    x : np.ndarray
        An input image .

    Returns
    -------
    corrected : np.ndarray
        The corrected input image, with the background removed.
    """
    maskh, maskw = estimate_mask(x)
    x = x.astype(np.float32)
    bg = estimate_background(x[maskh, maskw])
    corrected = x[maskh, maskw] - bg
    corrected = corrected - np.min(corrected)
    x[maskh, maskw] = corrected
    return x


def estimate_background(x: np.ndarray) -> np.ndarray:
    """Estimate background using a second order polynomial surface.

    Estimate the background of an image using a second-order polynomial surface
    assuming sparse signal in the image.  Essentially a massive least-squares
    fit of the image to the polynomial.

    Parameters
    ----------
    x : np.ndarray
        An input image which is to be used for estimating the background.

    Returns
    -------
    background_estimate : np.ndarray
        A second order polynomial surface representing the estimated background
        of the image.
    """

    # set up arrays for params and the output surface
    A = np.zeros((x.shape[0] * x.shape[1], 6))
    background_estimate = np.zeros((x.shape[1], x.shape[0]))

    u, v = np.meshgrid(
        np.arange(x.shape[1], dtype=np.float32),
        np.arange(x.shape[0], dtype=np.float32),
    )
    A[:, 0] = 1.0
    A[:, 1] = np.reshape(u, (x.shape[0] * x.shape[1],))
    A[:, 2] = np.reshape(v, (x.shape[0] * x.shape[1],))
    A[:, 3] = A[:, 1] * A[:, 1]
    A[:, 4] = A[:, 1] * A[:, 2]
    A[:, 5] = A[:, 2] * A[:, 2]

    # convert to a matrix
    A = np.matrix(A)

    # calculate the parameters
    k = np.linalg.inv(A.T * A) * A.T
    k = np.squeeze(np.array(np.dot(k, np.ravel(x))))

    # calculate the surface
    background_estimate = (
        k[0] + k[1] * u + k[2] * v + k[3] * u * u + k[4] * u * v + k[5] * v * v
    )
    return background_estimate


def estimate_mask(x: np.ndarray) -> Tuple[slice]:
    """Estimate the mask of a frame.

    Masking may occur when frame registration has been performed.

    Parameters
    ----------
    x : np.ndarray
        An input image which is to be used for estimating the background.

    Returns
    -------
    mask : tuple (2,)
        Slices representing the mask of the image.
    """
    if hasattr(x, "compute"):
        x = x.compute()
    nonzero = np.nonzero(x)
    sh = slice(np.min(nonzero[0]), np.max(nonzero[0]) + 1, 1)
    sw = slice(np.min(nonzero[1]), np.max(nonzero[1]) + 1, 1)
    return sh, sw


def parse_filename(filename: os.PathLike, fn_pattern = None) -> dict:
    """Parse an OctopusLite filename and retreive metadata from the file.

    Parameters
    ----------
    filename : PathLike
        The full path to a file to parse.
    fn_pattern : regex str
        Optional rewriting of default filename pattern regex

    Returns
    -------
    metadata : dict
        A dictionary containing the parsed metadata.
    """
    if fn_pattern:
        OCTOPUSLITE_FILEPATTERN = fn_pattern
    else:
        ### default fn pattern
        OCTOPUSLITE_FILEPATTERN = (
            # should be
            # TCZXY
            "img_p(?P<position>[0-9]+)_t(?P<time>[0-9]+)_z(?P<z>[0-9]+)_c(?P<channel>[0-9]+)"
        )

    pth, filename = os.path.split(filename)
    params = re.match(OCTOPUSLITE_FILEPATTERN, filename)

    # metadata = {
    #     "filename": filename,
    #     "channel": Channels(int(params.group("channel"))),
    #     "time": params.group("time"),
    #     "position": params.group("position"),
    #     "z": params.group("z"),
    #     # "timestamp": os.stat(filename).st_mtime,
    # }

    ### extract most of the metadata
    metadata = params.groupdict()
    ### convert the channel metadata from str to enumerated class
    metadata['channel'] = Channels(int(metadata['channel']))
    ### add filename to metadata
    metadata['filename'] = filename

    return metadata

def read_harmony_metadata(metadata_path: os.PathLike, assay_layout = False
    )-> pd.DataFrame:
    """
    Read the metadata from the Harmony software for the Opera Phenix microscope.
    Takes an input of the path to the metadata .xml file.
    Returns the metadata in a pandas dataframe format.
    If assay_layout is True then alternate xml format is anticipated, returning
    information about the assay layout of the experiment rather than the general
    organisation of image volume.
    """
    ### read xml metadata file
    print('Reading metadata XML file...')
    xml_data = open(metadata_path, 'r', encoding="utf-8-sig").read()
    root = ET.XML(xml_data)
    ### extraction procedure for image volume metadata
    if not assay_layout:
        ### extract the metadata from the xml file
        images_metadata = [child for child in root if "Images" in child.tag][0]
        ### create an empty list for storing individual image metadata
        metadata = list()
        ### iterate over every image entry extracting the metadata
        for image_metadata in tqdm(images_metadata, total = len(images_metadata),
                                    desc = 'Extracting HarmonyV5 metadata'):
            ### create empty dict to store single image metadata
            single_image_dict = dict()
            ### iterate over every metadata item in that image metadata
            for item in image_metadata:
                ### get column names from metadata
                col = item.tag.replace('{http://www.perkinelmer.com/PEHH/HarmonyV5}','')
                ### get metadata
                entry = item.text
                ### make dictionary out of metadata
                single_image_dict[col] = entry
            ### append that image metadata to list of all images
            metadata.append(single_image_dict)
    ### extraction procedure for assay layout metadata
    if assay_layout:
        metadata = dict()
        for branch in root:
            for subbranch in branch:
                if subbranch.text.strip() and subbranch.text.strip() != 'string':
                    col_name = subbranch.text
                    metadata[col_name] = dict()
                for subsubbranch in subbranch:
                    if 'Row' in subsubbranch.tag:
                        row = int(subsubbranch.text)
                    elif 'Col' in subsubbranch.tag and 'Color' not in subsubbranch.tag:
                        col = int(subsubbranch.text)
                    if 'Value' in subsubbranch.tag and subsubbranch.text != None:
                        val = subsubbranch.text
                        metadata[col_name][int(row), int(col)] = val

    ### create a dataframe out of all metadata
    df = pd.DataFrame(metadata)
    ### rename columns if assay layout
    # if assay_layout:
    #     columns = list(df.columns)
    #     columns.insert(0, 'Row, Col')
    #     df.rename(columns =)
    print('Extracting metadata complete!')
    return df


def crop_image(img: np.ndarray, crop: Tuple[int]) -> np.ndarray:
    """Crops a central window from an input image given a crop area size tuple

    Parameters
    ----------
    img : np.ndarray
        Input image.
    crop : tuple
        An tuple which is used to perform a centred crop on the
        image data.

    Returns
    -------
    img : np.ndarray
        The cropped image.

    """
    shape = img.shape
    dims = img.ndim
    cslice = lambda d: slice(
        int((shape[d] - crop[d]) // 2), int((shape[d] - crop[d]) // 2 + crop[d])
    )
    crops = tuple([cslice(d) for d in range(dims)])
    img = img[crops]

    return img
