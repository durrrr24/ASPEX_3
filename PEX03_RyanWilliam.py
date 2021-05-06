# Uncomment when using the realsense camera
import pyrealsense2.pyrealsense2 as rs  # For (most) Linux and Macs
#import pyrealsense2 as rs # For Windows
import numpy as np
import logging
import time
import datetime
import drone_lib
# import fg_camera_sim
import cv2
import imutils
import random
import logging
import traceback
import sys
import os
import glob
import shutil
from pathlib import Path

log = None  # logger instance
GRIPPER_OPEN = 1087
GRIPPER_CLOSED = 1940
gripper_state = GRIPPER_CLOSED # assume gripper is closed by default

IMG_SNAPSHOT_PATH = '/dev/drone_data/mission_data/RyanWilliam'
IMG_WRITE_RATE = 10  # write every 10 frames to disk...

# Various mission states:
# We start out in "seek" mode, if we think we have a target, we move to "confirm" mode,
# If target not confirmed, we move back to "seek" mode.
# Once a target is confirmed, we move to "target" mode.
# After positioning to target and calculating a drop point, we move to "deliver" mode
# After delivering package, we move to RTL to return home.
MISSION_MODE_SEEK = 0
MISSION_MODE_CONFIRM = 1
MISSION_MODE_TARGET = 2
MISSION_MODE_DELIVER = 4
MISSION_MODE_RTL = 8

# Tracks the state of the mission
mission_mode = MISSION_MODE_SEEK

# x,y center for 640x480 camera resolution.
FRAME_HORIZONTAL_CENTER = 320.0
FRAME_VERTICAL_CENTER = 240.0
FRAME_HEIGHT = int(480)
FRAME_WIDTH = int(640)

# random image
random_rbg = np.random.randint(0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3)).astype('uint8')


# Number of frames in a row we need to confirm a suspected target
REQUIRED_SIGHT_COUNT = 1  # must get 60 target sightings in a row to be sure of actual target

# Violet target
COLOR_RANGE_MIN = (110, 100, 75)
COLOR_RANGE_MAX = (160, 255, 255)


# Smallest object radius to consider (in pixels)
MIN_OBJ_RADIUS = 10

UPDATE_RATE = 1  # How many frames do we wait to execute on.

TARGET_RADIUS_MULTI = 1.7  # 1.5 x the radius of the target is considered a "good" landing if drone is inside of it.

# Font for use with the information window
font = cv2.FONT_HERSHEY_SIMPLEX

# variables
drone = None
counter = 0
direction1 = "unknown"
direction2 = "unknown"
inside_circle = False

# tracks number of attempts to re-acquire a target (if lost)
target_locate_attempts = 0


# info related to last (potential) target sighting
last_obj_lon = None
last_obj_lat = None
last_obj_alt = None
last_obj_heading = None
last_point = None  # center point in pixels

# thresholds for detect_annotate
CONF_THRESH, NMS_THRESH = 0.05, 0.3

# number of misses
tracker_misses = 0
tracker = None

# Uncomment below when using actual realsense camera
# Configure realsense camera stream
pipeline = rs.pipeline()
config = rs.config()

# net setup
net = None
classes = None
output_layers = None

# variables to modify in the field
target_circle_radius = 17
DRONE_ANGLE = 45
CENTERS_REQ = 1
# change to 4 for van
ASSUMED_WIDTH_MOD = 1
DISTANCE_SUB = 3
MAX_MISSES = 10
DROP_ALT = 3.2
VIRTUAL = False
range_finder_dist = None
tracker_class = "pedestrian"

def write_frame(frm_num, frame, path):
    frm = "{:06d}".format(int(frm_num))
    cv2.imwrite(f"{path}/frm_{frm}.png", frame)

def setup_net():
    global net, output_layers, classes

    in_weights = 'yolov4-tiny-custom_last.weights'
    in_config = 'yolov4-tiny-custom.cfg'
    name_file = 'custom.names'

    with open(name_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(in_config, in_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def release_grip(seconds=2):
    sec=1

    while sec <= seconds:
        override_gripper_state(GRIPPER_OPEN)
        time.sleep(1)
        sec += 1


def override_gripper_state(state=GRIPPER_CLOSED):
    global gripper_state
    gripper_state = state
    drone.channels.overrides['7'] = gripper_state


def backup_prev_experiment(path):
    if os.path.exists(path):
        if len(glob.glob(f'{path}/*')) > 0:
            time_stamp = time.time()
            shutil.move(os.path.normpath(path),
                        os.path.normpath(f'{path}_{time_stamp}'))

    Path(path).mkdir(parents=True, exist_ok=True)


def clear_path(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)


def start_camera_stream():
    logging.info("configuring rgb stream.")
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    logging.info("Starting camera streams...")
    profile = pipeline.start(config)


def get_cur_frame(attempts=5, flip_v=False):
    # Wait for a coherent pair of frames: depth and color
    tries = 0

    # This will capture the frames from the simulator.
    # If using an actual camera, comment out the two lines of
    # code below and replace with code that returns a single frame
    # from your camera.
    # image = fg_camera_sim.get_cur_frame()
    # return cv2.resize(image, (int(FRAME_HORIZONTAL_CENTER * 2), int(FRAME_VERTICAL_CENTER * 2)))
    rgb_frame = None

    # Code below can be used with the realsense camera...
    while tries <= attempts:
        try:
            frames = pipeline.wait_for_frames()
            rgb_frame = frames.get_color_frame()
            rgb_frame = np.asanyarray(rgb_frame.get_data())

            if flip_v:
                rgb_frame = cv2.flip(rgb_frame, 0)
            return rgb_frame
        except Exception:
            print(Exception)

        tries += 1
    if rgb_frame is None:
        return None


def get_ground_distance(hypotenuse):

    import math

    # Assuming we know the distance to object from the air
    # (the hypotenuse), we can calculate the ground distance
    # by using the simple formula of:
    # d^2 = hypotenuse^2 - height^2

    return math.sin(math.radians(DRONE_ANGLE)) * hypotenuse


def calc_new_location_to_target(from_lat, from_lon, heading, drone_distance):

    from geopy import distance
    from geopy import Point

    # given: current latitude, current longitude,
    #        heading = bearing in degrees,
    #        distance from current location (in meters)

    origin = Point(from_lat, from_lon)
    # destination = distance.distance(
    #     kilometers=(distance*.001)).destination(origin, heading)
    destination = distance.distance(
        kilometers=(drone_distance*.001)).destination(origin, heading)

    return destination.latitude, destination.longitude


def detect_objects(img):
    cv2.putText(img, 'detecting...', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,230), 2)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (192, 192), swapRB=False, crop=False)

    # blob = cv2.dnn.blobFromImage(
    #    cv2.resize(img, (416, 416)),
    #    0.007843, (416, 416), 127.5)

    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:

        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                # if it is what we're looking for
                # if classes[class_id] == "people" or classes[class_id] == "pedestrian":
                if classes[class_id] == tracker_class:

                    center_x, center_y, w, h = \
                        (detection[0:4] * np.array([FRAME_HORIZONTAL_CENTER * 2, FRAME_VERTICAL_CENTER * 2, FRAME_HORIZONTAL_CENTER * 2, FRAME_VERTICAL_CENTER * 2])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

    return b_boxes, confidences


def check_for_initial_target(frame):

    cv2.putText(frame, 'detecting...', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 230), 2)

    center = None
    radius = None
    x = y = None
    confidence = None

    # detect objects and return b_boxes, confidences and class_ids
    b_boxes, confidences = detect_objects(frame)
    # if we found an object
    if len(b_boxes) >= 1:
        max_confidence = max(confidences)
        max_index = confidences.index(max_confidence)
        # get the values from the best bounding box
        x, y, w, h = b_boxes[max_index]
        confidence = confidences[max_index]
        # calculate center of the object
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        radius = max(h, w) + 25
        center = (cx, cy)
        # return augmented bounding box
        return center, radius, (x, y), confidence

    else:
        return center, radius, (x, y), confidence


def determine_drone_actions(target_point, frame, target_sightings):
    global mission_mode, target_locate_attempts
    global direction1, direction2
    global target_circle_radius, inside_circle
    global last_obj_lon, last_obj_lat, last_obj_alt, last_obj_heading

    y_movement = 0.0
    x_movement = 0.0

    # Now, lets calculate our drone's actions according to what we have found...
    if target_point is not None:

        # dx = float(target_point's x position)- frame's horizontal center
        # dy = frame's vertical center -float(target_point's y position)
        dx = float(target_point[0]) - FRAME_HORIZONTAL_CENTER
        dy = FRAME_VERTICAL_CENTER - float(target_point[1])

        logging.info(f"Anticipated change in position towards target: dx={dx}, dy={dy}")


        # Check to see if we're inside our safe zone relative to target...
        if (int(target_point[0]) - FRAME_HORIZONTAL_CENTER) ** 2 \
                + (int(target_point[1]) - FRAME_VERTICAL_CENTER) ** 2 \
                < target_circle_radius ** 2:

            inside_circle = True
        else:
            inside_circle = False

        centered_hor = False
        centered_ver = False
        # if we're inside the circle return centered for calculations
        if inside_circle:
            drone_lib.log_activity("Drone centered", log=log)
            return True
        # if not move the drone
        else:
            if dx < 0:  # left
                # do what?  negative direction...
                if abs(dx) < 50:
                    x_movement = -0.2
                else:
                    x_movement = -0.5
                direction1 = "Need to go Left"
                # pass  # (REMOVE 'pass' when you have supplied actual code)
            if dx > 0:  # right
                # do what?  positive direction...
                if abs(dx) < 50:
                    x_movement = 0.2
                else:
                    x_movement = 0.5
                direction1 = "Need to go Right"
                # pass
            if dy < 0:  # back
                # do what?  positive direction...
                if abs(dy) < 50:
                    y_movement = 0.2
                else:
                    y_movement = 0.5
                direction2 = "Need to go Back"
                # pass
            if dy > 0:  # forward
                if abs(dy) < 50:
                    y_movement = -0.2
                else:
                    y_movement = -0.5
                direction2 = "Need to move Forward"
                # do what?  negative direction...
                # pass
            if abs(dx) < 7:  # if we are within 8 pixels, no need to make adjustment
                x_movement = 0.0
                direction1 = "Horizontal Center!"
                centered_hor = True
            if abs(dy) < 7:  # if we are within 8 pixels, no need to make adjustment
                y_movement = 0.0
                direction2 = "Vertical Center!"
                centered_ver = True
            # if within 7 pixels take calculations
            if centered_ver and centered_hor:
                drone_lib.log_activity("Drone centered", log=log)
                return True
            # else move the drone return not centered
            if x_movement > 0.0:
                drone_lib.small_move_right(drone, x_movement)
            else:
                drone_lib.small_move_left(drone, abs(x_movement))
            time.sleep(1)
            if y_movement > 0.0:
                drone_lib.small_move_back(drone, y_movement)
            else:
                drone_lib.small_move_forward(drone, abs(y_movement))
            time.sleep(1)
            return False



def setup_tracker(frame, bbox, radius):
    tracker = None
    tracker = cv2.TrackerCSRT_create()

    new_bbox = (bbox[0], bbox[1], radius, radius)
    success = tracker.init(frame, new_bbox)
    return success, tracker


def confirm_target_bbox(frame, b_box, net, classes, output_layers):
    center = None
    blank_image = random_rbg.copy()
    x = int(b_box[0])
    y = int(b_box[1])
    w = int(b_box[2])
    h = int(b_box[3])

    if w <= 0 or h <= 0:
        return False, None

    cx = int(x + w / 2)
    cy = int(y + h / 2)
    cr = int(max(w, h) / 2)

    cropped = frame[cy - cr:cy + cr, cx - cr:cx + cr]
    blank_image[y:y + cropped.shape[0], x:x + cropped.shape[1]] = cropped

    if VIRTUAL:
        cv2.imshow("cropped", blank_image)

    center, radius, (x, y), confidence = check_for_initial_target(blank_image)
    drone_lib.log_activity(f"Current confidence: {confidence}", log=log)

    x = int(b_box[0])
    y = int(b_box[1])
    center_x = x + int(b_box[2] / 2)
    center_y = y + int(b_box[3] / 2)
    if confidence is not None and confidence > .2:

        cv2.rectangle(frame, (x, y), (x + int(b_box[2]), y + int(b_box[3])),
                      (255, 0, 0), 2, 1)
        return True, (center_x, center_y)
    else:
        return False, (center_x, center_y)


def pixel_distance(widths):
    sum_distance = 0
    # assumed width of target in meters
    assumed_width = 1.0 * ASSUMED_WIDTH_MOD
    # adjust for car/van
    ratio = 0.01807

    # calculate ground distances for each width
    for width in widths:
        # formula for distance assumed width is 1 meter
        hypotenuse = (7.62 * assumed_width) / (width * ratio)
        distance = get_ground_distance(hypotenuse) - DISTANCE_SUB
        sum_distance += distance

    # calculate final distance in meters
    final_distance = sum_distance / len(widths)
    return final_distance


def range_finder_distance():
    distance = drone.rangefinder.distance
    if distance is None:
        return None
    ground_distance = get_ground_distance(distance)
    return ground_distance


def drop_off_sequence(lat, long, alt):
    # hover over to the point
    drone_lib.goto_point(drone, lat, long, 0.5, alt, log=log)
    # wait to stop moving
    time.sleep(1)

    # go down to 1.5 meters and release eggs
    drone_lib.log_activity(f"Lowering to: {DROP_ALT}", log=log)
    drone_lib.goto_point(drone, lat, long, 0.2, DROP_ALT, log=log)

    drone_lib.log_activity("Dropping off package", log=log)
    release_grip(2)


def landing_sequence(widths, alts, lats, longs, headings):

    # calculate final distance using pixel
    final_distance = pixel_distance(widths)

    # if pixel calculations fail
    if range_finder_dist is not None:
        if range_finder_dist > 0.0:
            final_distance = range_finder_dist

    # if both fail estimate
    if final_distance <= 0.0:
        final_distance = alts[0] * 1.25

    drone_lib.log_activity(f"final ground distance calculated: {final_distance}", log=log)

    # find last lat and long and heading
    lat = lats[len(lats) - 1]
    long = longs[len(longs) - 1]
    heading = headings[len(headings) - 1]

    drone_lib.log_activity("Going in for landing", log=log)


    # calculate final lat and long
    final_lat, final_long = calc_new_location_to_target(lat, long, heading, final_distance)

    # drop off sequence
    drop_off_sequence(final_lat, final_long, alts[0])

    # go home
    drone_lib.change_device_mode(drone, "RTL", log=log)


def conduct_mission():
    # Here, we will loop until we find a human target and deliver the care package,
    # or until the drone's flight plan completes (and we land).
    logging.info("Searching for target...")

    target_sightings = 0
    global counter, mission_mode, last_point, last_obj_lon, \
        last_obj_lat, last_obj_alt, \
        last_obj_heading, target_circle_radius, object_tracking, \
        tracker_misses, tracker, range_finder_dist

    object_tracking = False

    setup_net()

    logging.info("Starting camera feed...")
    start_camera_stream()
    center = None
    radius = None
    b_box = None
    confidence = None
    tracker = None
    tracker_misses = 0
    tracker_centers = 0
    widths = []
    alts = []
    lats = []
    longs = []
    headings = []

    while drone.armed:  # While the drone's mission is executing...

        # if manual override or mission is done
        if drone.mode == "RTL":
            mission_mode = MISSION_MODE_RTL
            logging.info("RTL mode activated.  Mission ended.")
            break

        # take a snapshot of current location
        location = drone.location.global_relative_frame
        last_lon = location.lon
        last_lat = location.lat
        last_alt = location.alt
        last_heading = drone.heading

        frame = get_cur_frame()

        # if rs camera is not working
        if frame is None:
            logging.info("Real Sense camera didn't load frames")
            drone_lib.change_device_mode(drone, "RTL", log=log)
            break

        frame_copy = frame.copy()
        cv2.circle(frame, (int(FRAME_HORIZONTAL_CENTER), int(FRAME_VERTICAL_CENTER)), int(target_circle_radius), (255, 255, 0), 2)
        # if not tracking
        if not object_tracking:
            # get initial object
            center, radius, b_box, confidence = check_for_initial_target(frame_copy)
            if confidence is not None and confidence > 0.2:
                # draw bounding box on things
                x = int(b_box[0])
                y = int(b_box[1])
                cv2.rectangle(frame, (x, y), (x + radius, y + radius), (20, 20, 230), 2)

                # send to object tracker and change to guided mode
                drone_lib.log_activity("Found target", log=log)
                drone_lib.change_device_mode(drone, "GUIDED", log=log)
                object_tracking = True
                success, tracker = setup_tracker(frame, b_box, radius)
        else:
            success, b_box = tracker.update(frame_copy)
            success, last_point = confirm_target_bbox(frame_copy, b_box, net, classes, output_layers)
            # record width
            width = b_box[2]
            # if still tracking, mode and adjust until centered
            if success:
                centered = determine_drone_actions(last_point, frame, target_sightings)

                # wait for the drone to finish movement
                if not centered:
                    if VIRTUAL:
                        # Display information in windowed frame:
                        cv2.putText(frame, direction1, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, direction2, (10, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # reset counter and alt and width records
                    tracker_centers = 0
                    widths.clear()
                    alts.clear()
                    lats.clear()
                    longs.clear()
                    headings.clear()
                else:
                    # calculate range finder distance
                    if not VIRTUAL:
                        range_finder_dist = range_finder_distance()

                    # wait for 7 consecutive centers and record info for new lat long
                    if VIRTUAL:
                        cv2.putText(frame, "centered", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    tracker_centers += 1
                    widths.append(width)
                    alts.append(last_alt)
                    lats.append(last_lat)
                    longs.append(last_lon)
                    headings.append(last_heading)
                # want 1 centers
                if tracker_centers >= CENTERS_REQ:
                    drone_lib.log_activity(f"widths calculated: {widths}", log=log)
                    drone_lib.log_activity(f"last alts: {alts}", log=log)
                    drone_lib.log_activity(f"last lats: {lats}", log=log)
                    drone_lib.log_activity(f"last longs: {lats}", log=log)
                    drone_lib.log_activity(f"last headings: {headings}", log=log)
                    landing_sequence(widths, alts, lats, longs, headings)

                tracker_misses = 0
            else:
                # reset centers and miss counters
                widths.clear()
                alts.clear()
                lats.clear()
                longs.clear()
                headings.clear()
                tracker_misses += 1
                tracker_centers = 0

            # if we miss the target 60 times in a row stop detecting send it back into auto
            if tracker_misses >= MAX_MISSES:
                drone_lib.log_activity("Lost target", log=log)
                drone_lib.change_device_mode(drone, "AUTO", log=log)
                time.sleep(2)
                object_tracking = False
                tracker = None
                tracker_misses = 0

        # if virtual show frame
        if VIRTUAL:
            cv2.imshow("Detecting", frame)

        # Now, show stats for informational purposes only
        if (counter % IMG_WRITE_RATE) == 0:
            #CAM
            write_frame(counter,frame,IMG_SNAPSHOT_PATH)
            #image_name = f"frm_{counter}.png"
            #cv2.imwrite(os.path.join(IMG_SNAPSHOT_PATH, image_name), frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            drone_lib.change_device_mode(drone, "RTL", log=log)
            break

        if mission_mode == MISSION_MODE_RTL:
            break  # mission is over.

        counter += 1

def main():
    global drone
    global log

    # Setup a log file for recording important activities during our session.
    log_file = time.strftime("KIRK_DURAN_PEX03_%Y%m%d-%H%M%S") + ".log"

    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    log = logging.getLogger(__name__)

    log.info("PEX 03 start.")

    # Connect to the autopilot

    #drone = drone_lib.connect_device("127.0.0.1:14550", log=log)
    drone = drone_lib.connect_device("/dev/ttyACM0", baud=115200, log=log)

    # Create a message listener using the decorator.
    print(f"Finder above ground: {drone.rangefinder.distance}")

    # Test latch - ensure open/close.
    # release_grip(2)

    # If the autopilot has no mission, terminate program
    drone.commands.download()
    time.sleep(1)

    log.info("Looking for mission to execute...")
    if drone.commands.count < 1:
        log.info("No mission to execute.")
        return

    # Arm the drone.
    drone_lib.arm_device(drone, log=log)

    # takeoff and climb 45 meters
    drone_lib.device_takeoff(drone, 20, log=log)

    try:
        # start mission
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        log.info("backing up old images...")

        # Backup any previous images and create new empty folder for current experiment.
        backup_prev_experiment(IMG_SNAPSHOT_PATH)

        # Now, look for target...
        conduct_mission()

        # Mission is over; disarm and disconnect.
        log.info("Disarming device...")
        drone.armed = False
        drone.close()
        log.info("End of demonstration.")
        pipeline.stop()
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise

main()
