"""Module containing functions for importing gerbers."""
import csv
import json
import subprocess
import os
import logging
from typing import List, Tuple
import sys
import re

import PIL.Image
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from gerber2ems.config import Config
from gerber2ems.constants import (
    GEOMETRY_DIR,
    UNIT,
    PIXEL_SIZE,
    BORDER_THICKNESS,
    STACKUP_FORMAT_VERSION,
)

import cv2
from triangle import triangulate

logger = logging.getLogger(__name__)


def process_gbrs_to_pngs():
    """Process all gerber files to PNG's.

    Finds edge cuts gerber as well as copper gerbers in `fab` directory.
    Processes copper gerbers into PNG's using edge_cuts for framing.
    Output is saved to `ems/geometry` folder
    """
    logger.info("Processing gerber files")

    files = os.listdir(os.path.join(os.getcwd(), "fab"))

    edge = next(filter(lambda name: "Edge_Cuts.gbr" in name, files), None)
    if edge is None:
        logger.error("No edge_cuts gerber found")
        sys.exit(1)

    layers = list(filter(lambda name: "_Cu.gbr" in name, files))
    if len(layers) == 0:
        logger.warning("No copper gerbers found")

    for name in layers:
        output = name.split("-")[-1].split(".")[0] + ".png"
        gbr_to_png(
            os.path.join(os.getcwd(), "fab", name),
            os.path.join(os.getcwd(), "fab", edge),
            os.path.join(os.getcwd(), GEOMETRY_DIR, output),
        )


def gbr_to_png(gerber_filename: str, edge_filename: str, output_filename: str) -> None:
    """Generate PNG from gerber file.

    Generates PNG of a gerber using gerbv.
    Edge cuts gerber is used to crop the image correctly.
    Output DPI is based on PIXEL_SIZE constant.
    """
    logger.debug("Generating PNG for %s", gerber_filename)
    not_cropped_name = f"{output_filename.split('.')[0]}_not_cropped.png"

    dpi = 1 / (PIXEL_SIZE * UNIT / 0.0254)
    if not dpi.is_integer():
        logger.warning("DPI is not an integer number: %f", dpi)

    gerbv_command = f"gerbv {gerber_filename} {edge_filename}"
    gerbv_command += " --background=#000000 --foreground=#ffffffff --foreground=#ffffffff"
    gerbv_command += f" -o {not_cropped_name}"
    gerbv_command += f" --dpi={dpi} --export=png -a"

    subprocess.call(gerbv_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    not_cropped_image = PIL.Image.open(not_cropped_name)

    # image_width, image_height = not_cropped_image.size
    cropped_image = not_cropped_image.crop(not_cropped_image.getbbox())
    cropped_image.save(output_filename)

    if not Config.get().arguments.debug:
        os.remove(not_cropped_name)


def get_dimensions(input_filename: str) -> Tuple[int, int]:
    """Return board dimensions based on png.

    Opens PNG found in `ems/geometry` directory,
    gets it's size and subtracts border thickness to get board dimensions
    """
    path = os.path.join(GEOMETRY_DIR, input_filename)
    image = PIL.Image.open(path)
    image_width, image_height = image.size
    height = image_height * PIXEL_SIZE - BORDER_THICKNESS
    width = image_width * PIXEL_SIZE - BORDER_THICKNESS
    logger.debug("Board dimensions read from file are: height:%f width:%f", height, width)
    return (width, height)


####


def image_to_mesh(path):

    im = cv2.transpose(cv2.imread(path))

    assert im is not None, "file could not be read, check with os.path.exists()"

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 20, 255, 0)
    _contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = _hierarchy[0]
    contours = [cv2.approxPolyDP(cnt, 2, True) for cnt in _contours]

    def find_point_inside_contour(i):
        contour = contours[i]

        x, y, w, h = cv2.boundingRect(contour)
        for px in range(x, x + w):
            for py in range(y, y + h):
                if cv2.pointPolygonTest(contour, (px, py), False) > 0:
                    children = get_children(i)
                    for child in children:
                        if cv2.pointPolygonTest(contours[child], (px, py), False) >= 0:
                            continue
                    return [px, py]
        raise Exception("Oh no 0 space contour")

    def pont_in_contour(i):
        # Calculate centroid of the contour (always inside if shape is convex)
        cnt = contours[i]
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        if cv2.pointPolygonTest(cnt, (cx, cy), False) <= 0:
            return find_point_inside_contour(i)

        children = get_children(i)
        for child in children:
            if cv2.pointPolygonTest(contours[child], (cx, cy), False) >= 0:
                return find_point_inside_contour(i)

        return [cx, cy]

    def is_hole(i):
        _, _, _, p = hierarchy[i]

        if p == -1:
            return False
        
        return not is_hole(p)
        
    def get_children(i):
        _, _, next, _ = hierarchy[i]

        children = []

        while next != -1:
            children.append(next)
            next, _, _, _ = hierarchy[next]

        return children

    # Plot cntrs result
    if Config.get().arguments.debug:
        cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
        plt.imshow(im)
        plt.show(block=True)

    triangles = []
    for i, cnt in enumerate(contours):
        if is_hole(i):
            continue
        
        pts = np.array(cnt).reshape(-1, 2).tolist()


        children = get_children(i)

        segments = [[i, (i+1) % len(pts)] for i in range(len(pts))]
        holes = []
        for child in children:
            shift = len(pts)

            _pts = np.array(contours[child]).reshape(-1, 2).tolist()
            _segments = [[shift + i, shift + ((i+1) % len(_pts))] for i in range(len(_pts))]

            pts.extend(_pts)
            segments.extend(_segments)
            holes.append(pont_in_contour(child))


        poly_dict = {
            "vertices": pts, 
            "segments": segments,
            **({"holes": holes} if holes else {})
        }
        
        # Perform Constrained Delaunay Triangulation
        result = triangulate(poly_dict, 'pa10000')  # 'p' ensures a constrained triangulation

        for tri in result['triangles']:
            ret = []
            for p in tri:
                x = result['vertices'][p][0]
                y = result['vertices'][p][1]
                point = np.array([x, y]) 
                ret.append(image_to_board_coordinates(point))
            triangles.append(ret)

    # Plot triangulated result
    if Config.get().arguments.debug:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # h, w = im.shape[:2]
        # aspect_ratio = h / w
        # ax.set_aspect(aspect_ratio, adjustable="box")

        for tri in triangles:
            xs = []
            ys = []
            for x, y in tri:
                xs.append(x)
                ys.append(y)
            ax.fill(ys, xs, facecolor='lightblue', edgecolor='black', alpha=0.7) # Transposed to match PCB coords

        plt.show(block=True)

    return np.array(triangles)

####

def get_triangles(input_filename: str) -> np.ndarray:
    """Triangulate image.

    Processes file from `ems/geometry`.
    Converts to grayscale, thresholds it to remove border
    and then uses Nanomesh to create a triangular mesh of the copper.
    Returns a list of triangles, where each triangle consists of coordinates for each vertex.
    """
    path = os.path.join(GEOMETRY_DIR, input_filename)
    return image_to_mesh(path)

def image_to_board_coordinates(point: np.ndarray) -> np.ndarray:
    """Transform point coordinates from image to board coordinates."""
    return (point * PIXEL_SIZE) - [BORDER_THICKNESS / 2, BORDER_THICKNESS / 2]


def get_vias() -> List[List[float]]:
    """Get via information from excellon file.

    Looks for excellon file in `fab` directory. Its filename should end with `-PTH.drl`
    It then processes it to find all vias.
    """
    files = os.listdir(os.path.join(os.getcwd(), "fab"))
    drill_filename = next(filter(lambda name: "-PTH.drl" in name, files), None)
    if drill_filename is None:
        logger.error("Couldn't find drill file")
        sys.exit(1)

    drills = {0: 0.0}  # Drills are numbered from 1. 0 is added as a "no drill" option
    current_drill = 0
    vias: List[List[float]] = []
    with open(os.path.join(os.getcwd(), "fab", drill_filename), "r", encoding="utf-8") as drill_file:
        for line in drill_file.readlines():
            # Regex for finding drill sizes (in mm)
            match = re.fullmatch("T([0-9]+)C([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                drills[int(match.group(1))] = float(match.group(2)) / 1000 / UNIT

            # Regex for finding drill switches (in mm)
            match = re.fullmatch("T([0-9]+)\\n", line)
            if match is not None:
                current_drill = int(match.group(1))

            # Regex for finding hole positions (in mm)
            match = re.fullmatch("X([0-9]+.[0-9]+)Y([0-9]+.[0-9]+)\\n", line)
            if match is not None:
                if current_drill in drills:
                    logger.debug(
                        f"Adding via at: X{float(match.group(1)) / 1000 / UNIT}Y{float(match.group(2)) / 1000 / UNIT}"
                    )
                    vias.append(
                        [
                            float(match.group(1)) / 1000 / UNIT,
                            float(match.group(2)) / 1000 / UNIT,
                            drills[current_drill],
                        ]
                    )
                else:
                    logger.warning("Drill file parsing failed. Drill with specifed number wasn't found")
    logger.debug("Found %d vias", len(vias))
    return vias


def import_stackup():
    """Import stackup information from `fab/stackup.json` file and load it into config object."""
    filename = "fab/stackup.json"
    with open(filename, "r", encoding="utf-8") as file:
        try:
            stackup = json.load(file)
        except json.JSONDecodeError as error:
            logger.error(
                "JSON decoding failed at %d:%d: %s",
                error.lineno,
                error.colno,
                error.msg,
            )
            sys.exit(1)
        ver = stackup["format_version"]
        if (
            ver is not None
            and ver.split(".")[0] == STACKUP_FORMAT_VERSION.split(".", maxsplit=1)[0]
            and ver.split(".")[1] >= STACKUP_FORMAT_VERSION.split(".", maxsplit=1)[1]
        ):
            Config.get().load_stackup(stackup)
        else:
            logger.error(
                "Stackup format (%s) is not supported (supported: %s)",
                ver,
                STACKUP_FORMAT_VERSION,
            )
            sys.exit()


def import_port_positions() -> None:
    """Import port positions from PnP .csv files.

    Looks for all PnP files in `fab` folder (files ending with `-pos.csv`)
    Parses them to find port footprints and inserts their position information to config object.
    """
    ports: List[Tuple[int, Tuple[float, float], float]] = []
    for filename in os.listdir(os.path.join(os.getcwd(), "fab")):
        if filename.endswith("-pos.csv"):
            ports += get_ports_from_file(os.path.join(os.getcwd(), "fab", filename))

    for number, position, direction in ports:
        if len(Config.get().ports) > number:
            port = Config.get().ports[number]
            if port.position is None:
                Config.get().ports[number].position = position
                Config.get().ports[number].direction = direction
            else:
                logger.warning(
                    "Port #%i is defined twice on the board. Ignoring the second instance",
                    number,
                )
    for index, port in enumerate(Config.get().ports):
        if port.position is None:
            logger.error("Port #%i is not defined on board. It will be skipped", index)


def get_ports_from_file(filename: str) -> List[Tuple[int, Tuple[float, float], float]]:
    """Parse pnp CSV file and return all ports in format (number, (x, y), direction)."""
    ports: List[Tuple[int, Tuple[float, float], float]] = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers
        for row in reader:
            if "Simulation_Port" in row[2]:
                number = int(row[0][2:])
                ports.append(
                    (
                        number - 1,
                        (float(row[3]) / 1000 / UNIT, float(row[4]) / 1000 / UNIT),
                        float(row[5]),
                    )
                )
                logging.debug("Found port #%i position in pos file", number)

    return ports
