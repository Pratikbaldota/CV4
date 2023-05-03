import cv2
import numpy as np

# Load the images
img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

#Display the img
#cv2.imshow('Image 1', img1)
#cv2.imshow('Image 2', img2)

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Display grayscale the img
#cv2.imshow('Grayscale 1', gray1)
#cv2.imshow('Grayscale 2', gray2)


# Detect corners in the first frame using the Harris corner detector
gray1 = np.float32(gray1)
dst1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
dst1 = cv2.dilate(dst1, None)
img1[dst1 > 0.01 * dst1.max()] = [0, 0, 255]

# Detect corners in the second frame using the Harris corner detector
gray2 = np.float32(gray2)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
dst2 = cv2.dilate(dst2, None)
img2[dst2 > 0.01 * dst2.max()] = [0, 0, 255]

# Display the images with detected corners
cv2.imshow('Corners Detected in Image 1', img1)
cv2.imshow('Corners Detected in Image 2', img2)

# Save the images with detected corners
cv2.imwrite('img1_corners.png', img1)
cv2.imwrite('img2_corners.png', img2)


# Find corners in the first frame using Harris corner detector
corners1 = cv2.goodFeaturesToTrack(gray1, 25, 0.01, 10)

# Create a mask image for drawing purposes
mask = np.zeros_like(img1)

# Draw the detected corners on the mask image
for corner in corners1:
    x, y = np.int0(corner.ravel())
    cv2.circle(mask, (y, x), 10, 255, -1)

# Display the image with detected corners
cv2.imshow('img1 with corners', cv2.add(img1, mask))

# Initialize tracker with first frame and detected corners
tracker = cv2.TrackerKCF_create()
x, y = np.int0(corners1[0].ravel())
w, h = 100, 100 # set the width and height of the bounding box
bbox = (x, y, w, h)
tracker.init(img1, bbox)

# Track the object in the second frame
success, bbox = tracker.update(img2)

if success:
    # Draw the tracked object
    x, y, w, h = np.int0(bbox)
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    print('Tracking failed')

# Display the tracked object in the second frame
cv2.imshow('Tracked object', img2)

# Save the images with detected corners
cv2.imwrite('corners_img1.png', cv2.add(img1, mask))
cv2.imwrite('tracked_obj_img2.png', img2)
Wait for user input and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

##############code working#######


fundamental matrix code
# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors for the images
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the keypoints
matches = bf.match(des1, des2)

# Sort the matches in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('Matches', img_matches)
# Save the modified image
cv2.imwrite('matches.png', img_matches)

# Calculate the fundamental matrix
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# Display the fundamental matrix
print('Fundamental matrix:\n', F)

# Wait for user input and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


#essential matrix
# Define the camera matrix
focal_length = 1000
principal_point_x = img1.shape[1] / 2
principal_point_y = img1.shape[0] / 2
K = np.array([[focal_length, 0, principal_point_x],
              [0, focal_length, principal_point_y],
              [0, 0, 1]])

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors for the images
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the keypoints
matches = bf.match(des1, des2)

# Sort the matches in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Calculate the essential matrix
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
E, mask = cv2.findEssentialMat(pts1, pts2, K)

# Display the essential matrix
print('Essential matrix:\n', E)

# Wait for user input and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

###ply file output
# Define the 3D coordinates of the points
points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Define the colors of the points (optional)
colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])

# Define the vertex list
vertex_list = []
for i in range(points.shape[0]):
    vertex = str(points[i, 0]) + ',' + str(points[i, 1]) + ',' + str(points[i, 2]) + ','
    if colors is not None:
        vertex += str(colors[i, 0]) + ' ' + str(colors[i, 1]) + ' ' + str(colors[i, 2])
    vertex_list.append(vertex)

# Write the PLY file
with open('output.ply', 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex ' + str(points.shape[0]) + '\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    if colors is not None:
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
    f.write('end_header\n')
    for vertex in vertex_list:
        f.write(vertex + '\n')

homography matrix

# Detect keypoints and extract descriptors using SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match the keypoints using a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches in order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Select the best 10% of the matches
num_matches = int(len(matches) * 0.1)
matches = matches[:num_matches]

# Extract the matched keypoints from the two images
points1 = np.zeros((num_matches, 2), dtype=np.float32)
points2 = np.zeros((num_matches, 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Calculate the homography matrix using RANSAC
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Display the homography matrix
print("Homography matrix:")
print(H)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors in both images
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Initialize brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Find matched points in both images
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

# Compute homography matrix using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply homography to img1 to align it with img2
h, w = img1.shape[:2]
aligned_img = cv2.warpPerspective(img1, M, (w, h))

# Display original images and aligned image
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.imshow('Aligned Image', aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
import numpy as np
import open3d as o3d

# Load the two images
img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Use ORB feature detector to find keypoints and descriptors in the images
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Use brute-force matcher to match the descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to filter out false matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get the coordinates of matched keypoints in both images
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Get the dimensions of the first image
h, w = gray1.shape

# Define the corners of the image in homogeneous coordinates
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(1, -1, 2)

# Transform the corners using the homography matrix
transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

# Write the 3D x, y, z coordinates separated by commas to an ASCII output PLY file
with open('output.ply', 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(len(transformed_corners)))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('end_header\n')
    for corner in transformed_corners:
        f.write('{:.6f}, {:.6f}, {:.6f}\n'.format(corner[0], corner[1], 0.0))

# Load the PLY file and visualize it using Open3D
pcd = o3d.io.read_point_cloud('output.ply', format='ply')
o3d.visualization.draw_geometries([pcd])
