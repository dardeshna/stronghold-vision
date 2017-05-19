# targeting.py
#
# Module for target detection and Head's Up Display (HUD)

"""
U-SHAPE REFERENCE

 -  6-5 <- 2" wide   2-1
 |  | |              | |
 |  | |              | |
 |  | |              | |
14" | |              | |
 |  | |              | |
 |  | 4--------------3 |
 -  7------------------0 (start)
    |------- 18"-------|
"""

from __future__ import division  # always use floating point division
import cv2
import numpy as np

""" Reference Target Contour """
# Load reference U shape and extract its contour
goal_img = cv2.imread('assets/goal.png', 0)
goal_contours, hierarchy = cv2.findContours(goal_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
goal_contour = goal_contours[0]


def find_best_match(frame):
    """
    This is essentially the goal detection function.

    Returns the contour that best matches the target shape.
    This is done by thresholding the image, extracting contours, approximating
    the contours as polygons. The polygonal contours are then filtered by
    vertice count and minimum area. Finally, the similarity of a contour is
    determined by the matchShapes() function, and the best_match variable is set
    if the similarity is lower than the prior similarity value.

    This function should not modify anything outside of its scope.
    """
    # Find outlines of white objects
    threshold = 200
    white = cv2.inRange(frame, (threshold, threshold, threshold), (255, 255, 255))  # threshold detection of white regions
    contours, heirarchy = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #find all contours in the image
    # Approximate outlines into polygons
    best_match = None # Variable to store best matching contour for U shape
    best_match_similarity = 1000 # Similarity of said contour to expected U shape. Defaults to an arbitrarily large number
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)  # smoothen the contours into simpler polygons
        # Filter through contours to detect a goal
        if cv2.contourArea(approx) > 1000 and len(approx) == 8:  # select contours with sufficient area and 8 vertices
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)  # draw the contour in red
            # test to see if this contour is the best match
            if check_match(approx):
                cv2.drawContours(frame, [approx], 0, (0, 128, 255), 2) # Draw U shapes in orange
                similarity = cv2.matchShapes(approx, goal_contour, 3, 0)
                if similarity < best_match_similarity:
                    best_match = approx
                    best_match_similarity = similarity

    return best_match


def check_match(contour):
    """
    Checks if the contour is the U shape by first finding the point that's
    furthest to the bottom right. Since the points in a contour are always
    ordered counterclockwise, we know which point in the contour is supposed
    to match up to each point of the U shape based on its position in the
    contour relative to the bottom-left point. Based off of this, we check to
    make sure that the distance between two consecutive points is correctly
    greater-than or less-than the distance between the two preceding points.
    If the distances change in the same pattern as they should in the U shape,
    then we consider the contour to be a match of the U shape.
    """
    # Get lower right point by finding point furthest from origin (top left)
    start_index = get_start_index(contour)

    # Check if the match could be good
    # should_be_less is a list of whether or not each distance should be less
    # than the previous distance to be a U shape.
    should_be_less = [True, False, False, True, True, False, False]
    prev_dist = -1
    for i in range(0, 8):
        point_a = contour[(i + start_index) % 8][0]
        point_b = contour[(i + start_index + 1) % 8][0]
        dist = (point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2

        if i > 0 and (dist < prev_dist) != should_be_less[i-1]:
            return False

        prev_dist = dist

    return True


def get_start_index(contour):
    """
    Returns the index of the point that's furthest to the bottom and the left
    (furthest from the origin). This point is referred to as the start point
    and the indices of other points will be made relative to it.
    """
    biggest_dist = -1
    start_index = -1
    for i in range(0, len(contour)):
        # Technically the square of the distance, but it doesn't matter since
        # we are only comparing the distances relative to each other
        dist = contour[i][0][0]**2 + contour[i][0][1]**2
        if dist > biggest_dist:
            biggest_dist = dist
            start_index = i

    return start_index


def get_nth_point(contour, n, start_index=-1):
    """
    Returns the nth point in a contour relative to the start index. If no start
    index is provided, it is calculated using the get_start_index function.
    """
    if start_index == -1:
        start_index = get_start_index(contour)

    return contour[(start_index + n) % len(contour)][0]


def target_center(target):
    """ Returns the top center point of a given target contour """
    left_pt = get_nth_point(target, 6)
    right_pt = get_nth_point(target, 1)
    return int((left_pt[0] + right_pt[0]) / 2), int((left_pt[1] + right_pt[1]) / 2)


def image_center(frame):
    """ Returns the center coordinate of the given frame """
    height, width = frame.shape[:2]
    return int(width/2), int(height/2)


