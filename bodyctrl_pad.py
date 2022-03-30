#!/usr/bin/env python

#================================================================
# Author  : Bruno JJE
# Date    : 04/2022
# License : GPL
#================================================================

import sys
import socket
import time
import struct

import cv2
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt


#==================================================================
# Config
#==================================================================

model_is_lite = True

use_delegate = False


use_server = True
do_debug_plot = False
do_plot_3d = False
display_skeleton = False

time_landmark_inference = False


#==================================================================
# Connect to server
#==================================================================
if use_server:

    # Get server IP address
    if len(sys.argv) < 2:
        TCP_IP_SERVER = input("Enter server IP address :")
    else:
        TCP_IP_SERVER = sys.argv[1]

    # Connect to server
    TCP_PORT_SERVER = 5007
    BUFFER_SIZE = 1024

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((TCP_IP_SERVER, TCP_PORT_SERVER))
    except:
        print("!!! ERROR: Server not found.")
        sys.exit()


#==================================================================
# Open webcam
#==================================================================
if len(sys.argv) < 3:
    CAP_INDEX = input("Enter video capture device index : ")
else:
    CAP_INDEX = sys.argv[2]

cap = cv2.VideoCapture(int(CAP_INDEX))
if not cap.isOpened():
    print("!!! ERROR: Can't open webcam device '%d'." % int(CAP_INDEX))
    sys.exit()


#==================================================================
# Functions
#==================================================================

def _normalize_color(color):
  return tuple(v / 255. for v in color)

def normalize_radians(angle):
    return angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi))


#==================================================================
# Define constants
#==================================================================

WHITE_COLOR = (224, 224, 224)
GRAY_COLOR = (200, 200, 200)
BLACK_COLOR = (0, 0, 0)
#RED_COLOR = (0, 0, 255)
#GREEN_COLOR = (0, 128, 0)
#BLUE_COLOR = (255, 0, 0)
LIGHTBLUE_COLOR = (255, 127, 0)
ORANGE_COLOR = (0, 127, 255)

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

BODY_CENTER = 33
BODY_TOP = 34

#LEFT_list = [LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, LEFT_EAR, MOUTH_LEFT, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX]
#
#RIGHT_list = [RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER, RIGHT_EAR, MOUTH_RIGHT, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX]
#

POSE_PAIRS_CENTER = [
                (9, 10),
                (11 ,12),
                (24,23) 
            ]

POSE_PAIRS_RIGHT = [
                (0,4),   (4,5),   (5,6),   (6, 8),
                (12,14), (14,16),
                (16,22), (16,18), (16,20), (18,20),
                (12,24), (24,26), (26,28),
                (28,32), (28,30), (30,32)
            ]

POSE_PAIRS_LEFT = [
                (0,1),   (1,2),   (2,3),   (3,7),
                (11,13), (13,15),
                (15,21), (15,17), (15,19), (19,17),
                (11,23), (23,25), (25,27),
                (27,29), (27,31), (29,31)
            ]


#==================================================================
# Create debug display
#==================================================================
if do_debug_plot:
    plt.ion()

    fig = plt.figure(figsize=(5, 5))
    ax_flat = plt.axes()
    ax_flat.scatter([], [], alpha=0.5)

    if do_plot_3d:
        # elevation: The elevation from which to view the plot.
        # azimuth: the azimuth angle to rotate the plot.
        elevation = 10
        azimuth = 10

        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

    plt.show()
    plt.pause(1)


#==================================================================
# Pose inference model
#==================================================================

# Inference model with TVM delagate
if use_delegate:

    import pyxir
    import tvm
    from tvm.contrib import graph_executor

    # load the pre-compiled module into memory
    lib = tvm.runtime.load_module("pose_landmark_full_pinto_float32/tvm_dpu_cpu.so")

    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))

    print(module.get_num_inputs())

# Inference model with CPU only
else:
    if model_is_lite:
        model_file_landmark = "pinto_blazepose_lite__model_float32.tflite"
    else:
        model_file_landmark = "pinto_blazepose_heavy__model_float32.tflite"

    interpreter_landmark = tf.lite.Interpreter(model_path=model_file_landmark)
    interpreter_landmark.allocate_tensors()

    print("\n\nLANDMARK DETAILS")
    landmark_input_details = interpreter_landmark.get_input_details()
    landmark_output_details = interpreter_landmark.get_output_details()
    print(landmark_input_details)
    print(landmark_output_details)

    print("\nLANDMARK INPUTS")
    for dd in landmark_input_details:
        print(dd['name'], dd['shape'])

    print("\nLANDMARK OUTPUTS")
    for dd in landmark_output_details:
        print(dd['name'], dd['shape'])

    #land_input_h, land_input_w = landmark_input_details[0]['shape'][1:3]


#==================================================================
# Init
#==================================================================
detect_rotation = 0

start_done = False
start_timer = -1
loop_timer = 0

first_detect = 0
got_landmark = 0

jump_armed = 0
timer_jump = 0
mode = 0
mode_toggle_count= 20
cmd = ""

time_old = time.time()


#==================================================================
# Main loop
#==================================================================
while cap.isOpened():

    #--------------------------------------------
    # Get an image from the webcam
    #--------------------------------------------

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    #--------------------------------------------
    # Squared input preparation
    #--------------------------------------------

    # Extend image to a square format with black border
    image_h, image_w, image_c = image.shape
    image_s = max(image_h, image_w)
    image_pad = abs(image_h-image_w)//2
    img_squared_bgr = np.zeros((image_s, image_s, image_c), np.uint8)
    if (image_h >= image_w):
        img_squared_bgr[:, image_pad:image_pad+image_w] = image
    else:
        img_squared_bgr[image_pad:image_pad+image_h, :] = image
    
    # Convert to RGB (cf. opencv image are in BGR format, but the model
    # requires an RGB format)
    img_squared = cv2.cvtColor(img_squared_bgr, cv2.COLOR_BGR2RGB)


    #--------------------------------------------
    # No detection : using landmark
    #--------------------------------------------

    # No landmarks are available yet (or we are restartingrestarting),
    # so the whole squared image will be used for the next inference
    if (start_timer > 0) or (got_landmark == 0):
        img_rotated = img_squared

        target_angle = math.pi * 0.5 # 90 = pi/2
        rotation = target_angle - math.atan2(-(0 - 0.5), 0)
        detect_rotation = normalize_radians(rotation)
        rect = [
                (0.5*image_s, 0.5*image_s),
                (1*image_s, 1*image_s),
                math.degrees(detect_rotation)
                ]

        box = cv2.boxPoints(rect)
        box = np.int0(box)

    # We use the previously obtained landmarks to define the area that will be used for the next inference
    else:

        # The obtained landmarks are not good enought to be used to define the area to process
        # for the next inference.
        # => reusing the previously defined area (if any)
        if landmark_poseflag_ptr[0] < 0.5:

            if first_detect == 0:
                img_rotated = img_squared
            else:
                img_rotated = cv2.warpPerspective(img_squared, M, (new_s, new_s))

        # The obtained landmarks are good enought to be used
        else:

            #---------------------------------------
            # Define the area for next inference
            #---------------------------------------

            # Bounding box size as double distance from center to top point with an additionnal 1.25 factor.
            body_size = math.sqrt((body_top_x-body_center_x)**2 + (body_top_y-body_center_y)**2) * 2
            body_size *= 1.25
            rect_w = body_size
            rect_h = body_size
            rect_x_center = body_center_x
            rect_y_center = body_center_y

            # Bounding box inclination
            target_angle = math.pi * 0.5 # 90 = pi/2
            rotation = target_angle - math.atan2(-(body_top_y - body_center_y), body_top_x - body_center_x)
            detect_rotation = normalize_radians(rotation)

            rect = [
                    (body_center_x*image_s, body_center_y*image_s),
                    (body_size*image_s, body_size*image_s),
                    math.degrees(detect_rotation)
                    ]

            box = cv2.boxPoints(rect)
            box = np.int0(box)


            # Draw body center and top on 'img_squared_bgr'
            tmp_x = int(body_center_x*image_s)
            tmp_y = int(body_center_y*image_s)
            cv2.circle(img_squared_bgr, (tmp_x, tmp_y), 8, (0, 255, 255), 2)
            tmp_x = int(body_top_x*image_s)
            tmp_y = int(body_top_y*image_s)
            cv2.circle(img_squared_bgr, (tmp_x, tmp_y), 8, (0, 255, 255), 4)

            # Draw squared box selected for next inference on 'img_squared_bgr'
            cv2.drawContours(img_squared_bgr, [box], 0, (0, 0, 255), 2)


            #---------------------------------------
            # Extract the area for next inference
            #---------------------------------------

            #    1--------2
            #    |        |
            #    |        |
            #    0--------3

            # get squared box size
            rect_s = body_size*image_s
            new_s = int(rect_s)

            src_pts = box.astype("float32")

            # Coordinate of the points in box points after it has been straightened
            dst_pts = np.array([
                                [0, rect_s-1],
                                [0, 0],
                                [rect_s-1, 0],
                                [rect_s-1, rect_s-1],
                                ], dtype="float32")

            # Perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Directly warp the rotated rectangle to get the straightened rectangle
            img_rotated = cv2.warpPerspective(img_squared, M, (new_s, new_s))


    #--------------------------------------------
    # landmark input preparation
    #--------------------------------------------

    # Resize image to the 256x256 format awaited by the landmark model 
    img_rotated_small = cv2.resize(img_rotated, (256,256), interpolation = cv2.INTER_AREA)

    #cv2.imshow('img_rotated_small', cv2.flip(img_rotated_small, 1))

    # reshape
    input_data_landmark = np.array(img_rotated_small.reshape([1, 256, 256, 3]), dtype=np.float32)

    # normalize [0, 255] to [-1, 1].
    mmax =  1
    mmin = -1
    input_data_landmark = input_data_landmark * (mmax-mmin)/255.0 + mmin


    #--------------------------------------------
    # Inference landmark
    #--------------------------------------------

    if time_landmark_inference:
        time_landmark_start = time.time()

    # landmark inference with TVM delegate
    if use_delegate:

        # tvm model with float input, no quantization required
        tvm_input_data = input_data_landmark

        # load input
        input_data = tvm.nd.array(tvm_input_data)
        module.set_input("input_1:0", input_data)
        # run inference
        module.run()

        # Get output
        #nb_out = module.get_num_outputs()
        #print("nb output =", nb_out)

        # delegate full float32 order
        landmark_ptr = module.get_output(0).numpy()
        landmark_world_ptr = module.get_output(1).numpy()
        landmark_heatmap_ptr = module.get_output(2).numpy()
        landmark_segmentation_ptr = module.get_output(3).numpy()
        landmark_poseflag_ptr = module.get_output(4).numpy()

        # tvm model with float output, no dequantization required

    # landmark inference with CPU
    else:

        # load input
        interpreter_landmark.set_tensor(landmark_input_details[0]['index'], input_data_landmark)
        # run inference
        interpreter_landmark.invoke()

        # Get output
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        if model_is_lite:
            # pinto lite model float32
            landmark_ptr = interpreter_landmark.get_tensor(landmark_output_details[0]['index'])[0]
            landmark_world_ptr = interpreter_landmark.get_tensor(landmark_output_details[1]['index'])[0]
            landmark_segmentation_ptr = interpreter_landmark.get_tensor(landmark_output_details[2]['index'])[0]
            landmark_poseflag_ptr = interpreter_landmark.get_tensor(landmark_output_details[3]['index'])[0]
            landmark_heatmap_ptr = interpreter_landmark.get_tensor(landmark_output_details[4]['index'])[0]
        else:
            # pinto heavy model float32
            landmark_ptr = interpreter_landmark.get_tensor(landmark_output_details[0]['index'])[0]
            landmark_world_ptr = interpreter_landmark.get_tensor(landmark_output_details[1]['index'])[0]
            landmark_segmentation_ptr = interpreter_landmark.get_tensor(landmark_output_details[2]['index'])[0]
            landmark_heatmap_ptr = interpreter_landmark.get_tensor(landmark_output_details[3]['index'])[0]
            landmark_poseflag_ptr = interpreter_landmark.get_tensor(landmark_output_details[4]['index'])[0]

    if time_landmark_inference:
        time_landmark = time.time() - time_landmark_start
        print("time_landmark: %.3f ms" % (time_landmark*1000))


    landmark_ptr = landmark_ptr.reshape([39, 5])
    landmark_world_ptr = landmark_world_ptr.reshape([39, 3])



    #------------------------------------------------------------------
    # (Re)Start control
    #------------------------------------------------------------------
    # Press 'space' to start a countdown, and have time to position
    # yourself in front of the webcam.
    # Can also be used to re-initialize the landmark detection if it
    # get lost.
    #------------------------------------------------------------------
    if cv2.waitKey(5) & 0xFF == 32: # space bar
        first_detect = 0
        start_timer = 10
        loop_timer = 0

    if loop_timer > 0:
        loop_timer -= 1
    elif start_timer > -1:
        start_timer -= 1
        print(start_timer)
        loop_timer = 2

    if start_timer == 0:
        print("Start !")
        start_done = True


    #------------------------------------------------------------------
    # Process landmark for 'body control'
    #------------------------------------------------------------------
    if landmark_poseflag_ptr[0] > 0.5:

        got_landmark = 1

        #--------------------------------------------------------------
        # Correct inference area rotation on image landmark coordinate
        #--------------------------------------------------------------

        back_src = np.array([
                            [0, 0],
                            [1, 0],
                            [1, 1],
                            ], dtype="float32")

        back_dst = box[1:].astype("float32") / image_s
        #src_pts = box.astype("float32") / image_s * img_squared.shape[0]

        mat = cv2.getAffineTransform(back_src, back_dst)
        lm_xy = np.expand_dims(landmark_ptr[:39,:2]/256, axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)) 

        #--------------------------------------------------------------
        # Correct inference area rotation on world landmark coordinate
        #--------------------------------------------------------------

        sin_rot = math.sin(detect_rotation)
        cos_rot = math.cos(detect_rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        landmark_world_ptr[:,:2] = np.dot(landmark_world_ptr[:,:2], rot_m)


        #--------------------------------------------------------------
        # Extract body center and top
        # (will be used to define the areao to use for next inference)
        #--------------------------------------------------------------

        body_center_x, body_center_y = lm_xy[BODY_CENTER][0:2]
        body_top_x, body_top_y = lm_xy[BODY_TOP][0:2]



        #--------------------------------------------------------------
        # Extract points used for 'body control'
        #--------------------------------------------------------------
        # With camera set horizontaly
        # x: side (right and left) (positive is right)
        # y: height (down and up) (positive is down)
        # z: depth (backward, forward) (positive is back, ie away from camera) 
        #--------------------------------------------------------------

        # shoulders : right, left, middle
        landmark_r = landmark_world_ptr[RIGHT_SHOULDER]
        landmark_l = landmark_world_ptr[LEFT_SHOULDER]
        x = [landmark_r[0], landmark_l[0], (landmark_r[0]+landmark_l[0])/2]
        y = [landmark_r[1], landmark_l[1], (landmark_r[1]+landmark_l[1])/2]
        z = [landmark_r[2], landmark_l[2], (landmark_r[2]+landmark_l[2])/2]

        # hips : right, left, middle
        landmark_rh = landmark_world_ptr[RIGHT_HIP]
        landmark_lh = landmark_world_ptr[LEFT_HIP]
        xh = [landmark_rh[0], landmark_lh[0], (landmark_rh[0]+landmark_lh[0])/2]
        yh = [landmark_rh[1], landmark_lh[1], (landmark_rh[1]+landmark_lh[1])/2]
        zh = [landmark_rh[2], landmark_lh[2], (landmark_rh[2]+landmark_lh[2])/2]

        # wrist : right, left, middle
        landmark_rw = landmark_world_ptr[RIGHT_WRIST]
        landmark_lw = landmark_world_ptr[LEFT_WRIST]
        xw = [landmark_rw[0], landmark_lw[0], (landmark_rw[0]+landmark_lw[0])/2]
        yw = [landmark_rw[1], landmark_lw[1], (landmark_rw[1]+landmark_lw[1])/2]
        zw = [landmark_rw[2], landmark_lw[2], (landmark_rw[2]+landmark_lw[2])/2]

        # ankle : right, left, middle
        landmark_ra = landmark_world_ptr[RIGHT_ANKLE]
        landmark_la = landmark_world_ptr[LEFT_ANKLE]
        xa = [landmark_ra[0], landmark_la[0], (landmark_ra[0]+landmark_la[0])/2]
        ya = [landmark_ra[1], landmark_la[1], (landmark_ra[1]+landmark_la[1])/2]
        za = [landmark_ra[2], landmark_la[2], (landmark_ra[2]+landmark_la[2])/2]

        # nose
        landmark_rn = landmark_world_ptr[NOSE]
        xn = [landmark_rn[0]]
        yn = [landmark_rn[1]]
        zn = [landmark_rn[2]]


        # Vectors
        rv_right = [xh[0], yh[0], zh[0]]
        rv_up = [x[2], y[2], z[2]]
        rv_r_hand = [xw[0], yw[0], zw[0]]
        rv_l_hand = [xw[1], yw[1], zw[1]]
        #rv_r_foot = [xa[0]-xh[2], ya[0]-yh[2], za[0]-zh[2]]
        #rv_l_foot = [xa[1]-xh[2], ya[1]-yh[2], za[1]-zh[2]]
        rv_r_foot = [xa[0], ya[0], za[0]]
        rv_l_foot = [xa[1], ya[1], za[1]]
        #rv_target = [xw[2]-xn[0], yw[2]-yn[0], zw[2]-zn[0]] # nose to middle of the wrist
        #rv_target = [xw[2]-x[2], yw[2]-y[2], zw[2]-z[2]] # middle of the shoulder to middle of the wrist
        #rv_target = [xw[2]-x[2]/2, yw[2]-y[2]/2, zw[2]-z[2]/2] # middle of the torso to middle of the wrist
        rv_target = [xw[2]-x[2]*3/4, yw[2]-y[2]*3/4, zw[2]-z[2]*3/4] # upper part of the torso to middle of the wrist
        rv_noose = [xn[0], yn[0], zn[0]]


        #--------------------------------------------------------------
        # Determine commands
        #--------------------------------------------------------------
        if start_done:

            cmd = ""

            if do_debug_plot:
                ax_flat.clear()
                ax_flat.set_xlim(-1, 1)
                ax_flat.set_ylim(-1, 1)
                #ax_flat.scatter([0, rv_up[1]], [0, -rv_up[2]], s=[200, 200], marker='+', c=[_normalize_color(BLACK_COLOR[::-1]), _normalize_color(LIGHTBLUE_COLOR[::-1])]) #, alpha=0.5)
                ax_flat.scatter([0, -x[2]], [0, -z[2]], s=[200, 200], marker='+', c=[_normalize_color(BLACK_COLOR[::-1]), _normalize_color(LIGHTBLUE_COLOR[::-1])]) #, alpha=0.5)


                ax_flat.scatter([rv_l_foot[0], rv_r_foot[0]], [-rv_l_foot[2], -rv_r_foot[2]], s=[200, 200], marker='o', c=[_normalize_color(BLACK_COLOR[::-1]), _normalize_color(LIGHTBLUE_COLOR[::-1])]) #, alpha=0.5)

                #ax_flat.scatter([0, rv_noose[0]], [0, -rv_noose[2]], s=[200, 200], marker='x', c=[_normalize_color(BLACK_COLOR[::-1]), _normalize_color(ORANGE_COLOR[::-1])]) #, alpha=0.5)

                x1, y1 = [-1, 1], [0.25, 0.25]
                x2, y2 = [-1, 1], [0.10, 0.10]
                x3, y3 = [-0.07, -0.07], [-1, 1]
                x4, y4 = [0.07, 0.07], [-1, 1]
                plt.plot(x1, y1, x2, y2, x3, y3, x4, y4)


            # right/left command
            if -x[2] > 0.1:
                if do_debug_plot:
                    plt.text(0.8, 0, "right")
                cmd += "R"
            elif -x[2] < -0.1:
                if do_debug_plot:
                    plt.text(-0.8, 0, "left")
                cmd += "L"
            else:
                cmd += "|"

            # wrist distance
            rv_rl_hand = np.array(rv_r_hand) - np.array(rv_l_hand)
            wrist_distance = np.linalg.norm(rv_rl_hand)
            if do_debug_plot:
                plt.text(-0.5, -0.5, "dw=%.2f" % wrist_distance, fontsize=16)

            # ankle distance
            rv_rl_foot = np.array(rv_r_foot) - np.array(rv_l_foot)
            ankle_distance = np.linalg.norm(rv_rl_foot)
            if do_debug_plot:
                plt.text(-0.5, -0.7, "da=%.2f" % ankle_distance, fontsize=16)

            # forward/backward command
            if ankle_distance > 0.55:
                if do_debug_plot:
                    plt.text(0, 0.8, "FORWARD")
                cmd += "F"
            elif ankle_distance > 0.4:
                if do_debug_plot:
                    plt.text(0, 0.8, "forward")
                cmd += "f"
            elif ankle_distance < 0.25:
                if do_debug_plot:
                    plt.text(0, -0.8, "backward")
                cmd += "B"
            else:
                cmd += "-"

            # arms length
            rv_r_arm = np.array(rv_r_hand) - np.array(rv_up)
            rv_l_arm = np.array(rv_l_hand) - np.array(rv_up)
            arm_r_length = np.linalg.norm(rv_r_arm)
            arm_l_length = np.linalg.norm(rv_l_arm)
            arm_length = (arm_r_length + arm_l_length) / 2 
            if do_debug_plot:
                plt.text(0.5, -0.5, "%.2f" % arm_length, fontsize=16)

            # legs length
            rv_r_leg = np.array(rv_r_foot) - np.array([xh[2], yh[2], zh[2]])
            rv_l_leg = np.array(rv_l_foot) - np.array([xh[2], yh[2], zh[2]])
            leg_r_length = np.linalg.norm(rv_r_leg)
            leg_l_length = np.linalg.norm(rv_l_leg)
            leg_length = (leg_r_length + leg_l_length) / 2 
            if do_debug_plot:
                plt.text(-0.5, 0.5, "%.2f" % leg_length, fontsize=16)

            # bodycontrol on/off
            if wrist_distance > 0.7 and ankle_distance > 0.55:
                if mode_toggle_count > 0:
                    mode_toggle_count -= 1
                if mode_toggle_count == 1:
                    mode = (mode + 1) % 2
            else:
                mode_toggle_count = 20

            if do_debug_plot:
                if mode==0:
                    fig.patch.set_facecolor((1, 1, 1))
                else:
                    fig.patch.set_facecolor((0, 0.8, 0))

            # aiming command
            if do_debug_plot:
                ax_flat.scatter([0, -rv_target[0]], [0, -rv_target[1]], s=[200, 200], marker='x', c=[_normalize_color(BLACK_COLOR[::-1]), _normalize_color(ORANGE_COLOR[::-1])]) #, alpha=0.5)
            
            if -rv_target[0] > 0.1:  cmd += "r"
            if -rv_target[0] < -0.1: cmd += "l"
            if -rv_target[1] > 0.1-0.1:  cmd += "u"
            if -rv_target[1] < -0.1-0.1: cmd += "d"

            if -rv_target[0] > 0.2:  cmd += "r"
            if -rv_target[0] < -0.2: cmd += "l"
            if -rv_target[1] > 0.2-0.1:  cmd += "u"
            if -rv_target[1] < -0.2-0.1: cmd += "d"

            if -rv_target[0] > 0.3:  cmd += "r"
            if -rv_target[0] < -0.3: cmd += "l"
            if -rv_target[1] > 0.3-0.1:  cmd += "u"
            if -rv_target[1] < -0.3-0.1: cmd += "d"


            # fire command
            if wrist_distance < 0.4:
                if do_debug_plot:
                    plt.text(0.5, -0.8, "FIRE", fontsize=16)
                cmd += "S" # shot
            else:
                cmd += "." # no shot

            # crouch command
            if leg_length < 0.4:
                cmd += "C"
                if do_debug_plot:
                    plt.text(-0.5, 0.8, "CRUNCH", fontsize=16)
                    ax_flat.set_facecolor((0.7, 0.7, 0.7))
            else:
                if do_debug_plot:
                    ax_flat.set_facecolor((1.0, 1.0, 1.0))
            
            # jump command
            if timer_jump > 0:
                timer_jump -= 1
                if do_debug_plot:
                    ax_flat.set_facecolor("yellow")
            else:
                if do_debug_plot:
                    ax_flat.set_facecolor((1.0, 1.0, 1.0))

            if leg_length < 0.55:
                jump_armed = 5

            if leg_length > 0.65 and jump_armed > 0:
                jump_armed = 0
                cmd += "J"
                if do_debug_plot:
                    plt.text(-0.5, 0.9, "!!! JUMP !!!", fontsize=16)
                    ax_flat.set_facecolor("yellow")
                timer_jump = 10
            elif jump_armed > 0:
                jump_armed -= 1
                if do_debug_plot:
                    plt.text(-0.5, 0.9, "JUMP ARMED", fontsize=16)

            # fps computation
            time_new = time.time()
            delta = time_new - time_old
            time_old = time_new
            if do_debug_plot:
                plt.text(0.1, 0, "fps=%d"%(int(1/delta)), fontsize=16)

            # draw debug plot
            if do_debug_plot:
                plt.draw()
                plt.pause(0.001)

            print(mode, "DELAY =", int(delta*1000), "ms    FPS =", int(1/delta), "   cmd =", cmd)


        #--------------------------------------------------------------
        # Transfert commands to server
        #--------------------------------------------------------------
        if start_done and use_server and mode:
            packer = struct.Struct('16c')
            packed_data = packer.pack(*[bytes(c, "ascii") for c in cmd.ljust(16)])
            s.sendall(packed_data)


        #--------------------------------------------------------------
        # Display skeleton on image
        #--------------------------------------------------------------

        #kpts = []
        #for i in range(39):
        #    kpts.append((int(landmark_ptr[i][0]),int(landmark_ptr[i][1])))
        #
        ##for pair in POSE_PAIRS:
        ##    cv2.line(img_rotated_small, kpts[pair[0]], kpts[pair[1]], (0, 255, 0), thickness=2)
        #for pair in POSE_PAIRS_CENTER:
        #    cv2.line(img_rotated_small, kpts[pair[0]], kpts[pair[1]], WHITE_COLOR, thickness=2)
        #for pair in POSE_PAIRS_RIGHT:
        #    cv2.line(img_rotated_small, kpts[pair[0]], kpts[pair[1]], LIGHTBLUE_COLOR, thickness=2)
        #for pair in POSE_PAIRS_LEFT:
        #    cv2.line(img_rotated_small, kpts[pair[0]], kpts[pair[1]], ORANGE_COLOR, thickness=2)

        if (display_skeleton) and (got_landmark == 1):
            preds = lm_xy*image_s
            kpts = []
            for i in range(39):
                kpts.append((int(preds[i][0]),int(preds[i][1])))

            for pair in POSE_PAIRS_CENTER:
                cv2.line(img_squared_bgr, kpts[pair[0]], kpts[pair[1]], WHITE_COLOR, thickness=3)
            for pair in POSE_PAIRS_RIGHT:
                cv2.line(img_squared_bgr, kpts[pair[0]], kpts[pair[1]], LIGHTBLUE_COLOR, thickness=3)
            for pair in POSE_PAIRS_LEFT:
                cv2.line(img_squared_bgr, kpts[pair[0]], kpts[pair[1]], ORANGE_COLOR, thickness=3)

            # body center and top
            cv2.line(img_squared_bgr, kpts[33], kpts[34], (0, 255, 255), thickness=6)

            for i in range(33):
                cv2.circle(img_squared_bgr, (int(lm_xy[i][0]*image_s),int(lm_xy[i][1]*image_s)), 4, (255,255,255), 2)

            # wrist and hand center with no z (for holistic hand tracking)
            for i in range(35, 39):
                cv2.circle(img_squared_bgr, (int(lm_xy[i][0]*image_s),int(lm_xy[i][1]*image_s)), 10, (0,255,0), 2)

        cv2.imshow('img_squared_bgr', cv2.flip(img_squared_bgr, 1))


        #--------------------------------------------------------------
        # Display skeleton in 3d
        #--------------------------------------------------------------
        if do_plot_3d:
            ax.clear()

            for pair in POSE_PAIRS_CENTER:
                ax.plot3D(
                    xs=[landmark_world_ptr[pair[0]][0], landmark_world_ptr[pair[1]][0]],
                    ys=[landmark_world_ptr[pair[0]][2], landmark_world_ptr[pair[1]][2]],
                    zs=[-landmark_world_ptr[pair[0]][1], -landmark_world_ptr[pair[1]][1]],
                    #color=[(1.,1.,1.), (1.,1.,1.)],
                    color='gray',
                    linewidth=2)
            for pair in POSE_PAIRS_RIGHT:
                ax.plot3D(
                    xs=[landmark_world_ptr[pair[0]][0], landmark_world_ptr[pair[1]][0]],
                    ys=[landmark_world_ptr[pair[0]][2], landmark_world_ptr[pair[1]][2]],
                    zs=[-landmark_world_ptr[pair[0]][1], -landmark_world_ptr[pair[1]][1]],
                    #color=[(1,0.5,0)],
                    color='blue',
                    linewidth=2)
            for pair in POSE_PAIRS_LEFT:
                ax.plot3D(
                    xs=[landmark_world_ptr[pair[0]][0], landmark_world_ptr[pair[1]][0]],
                    ys=[landmark_world_ptr[pair[0]][2], landmark_world_ptr[pair[1]][2]],
                    zs=[-landmark_world_ptr[pair[0]][1], -landmark_world_ptr[pair[1]][1]],
                    #color=[(0,0.5,1)],
                    color='orange',
                    linewidth=2)

            for i in range(33):
                ax.scatter3D(
                    xs=[landmark_world_ptr[i][0]],
                    ys=[landmark_world_ptr[i][2]],
                    zs=[-landmark_world_ptr[i][1]],
                    color=[(1,0,1)],
                    linewidth=2)

            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.draw()
            plt.pause(0.001)



    #--------------------------------------------------------------
    # Check ESC key press for end
    #--------------------------------------------------------------
    if cv2.waitKey(5) & 0xFF == 27:
      break


#--------------------------------------------------------------
# End
#--------------------------------------------------------------
cap.release()

if use_server:
    s.close()

