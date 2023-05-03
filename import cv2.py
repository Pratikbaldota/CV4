# import cv2
# import numpy as np

# # Load the two input images
# img1 = cv2.imread('img1.png')
# img2 = cv2.imread('img2.png')

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Define the Harris corner detector parameters
# corner_params = dict( maxCorners = 100,
#                       qualityLevel = 0.3,
#                       minDistance = 7,
#                       blockSize = 7 )

# # Find the corners in the first image using the Harris corner detector
# corners1 = cv2.goodFeaturesToTrack(gray1, **corner_params)
# corners1 = np.int0(corners1)

# # Reshape the corners1 array to have shape (N, 1, 2)
# corners1 = corners1.reshape(-1, 1, 2)

# # Track the corners in the second image using the Lucas-Kanade method
# corners2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners1, None)

# # Convert the corners1 and corners2 arrays to the appropriate data type
# corners1 = corners1.astype(np.float32)
# corners2 = corners2.astype(np.float32)

# # Keep only the corners that were successfully tracked
# good_corners1 = corners1[status==1]
# good_corners2 = corners2[status==1]

# # Estimate the Fundamental matrix from the tracked corners
# F, mask = cv2.findFundamentalMat(good_corners1, good_corners2, cv2.FM_RANSAC)

# # Compute the Essential matrix from the Fundamental matrix and the camera matrices
# K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]]) # Intrinsic parameters of the camera
# E = np.dot(np.dot(K.T, F), K)

# # Decompose the Essential matrix into the relative camera motion and the 3D scene structure
# U, S, Vt = np.linalg.svd(E)
# W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
# R1 = np.dot(np.dot(U, W), Vt)
# R2 = np.dot(np.dot(U, W.T), Vt)
# t1 = U[:,2]
# t2 = -U[:,2]

# # Compute the 3D scene structure using the Factorization method
# n_pts = good_corners1.shape[0]

# # Normalize the image coordinates
# x1 = np.hstack((good_corners1, np.ones((n_pts, 1)))).T
# x2 = np.hstack((good_corners2, np.ones((n_pts, 1)))).T
# x1n = np.dot(np.linalg.inv(K), x1)
# x2n = np.dot(np.linalg.inv(K), x2)

# # Construct the measurement matrix
# M = np.zeros((2*n_pts, 4))
# for i in range(n_pts):
#     M[2*i:2*i+2,:] = np.kron(x1n[:,i], x2n[:,i]).reshape((2,4))

# # Perform SVD on the measurement matrix
# U, S, Vt = np.linalg.svd(M)

# # Extract the 3D scene structure from the factorization matrix
# X = Vt[-1,:].reshape((4, n_pts))
# X = X[:3,:] / X[3,:]

# # Compute the reprojection error
# P1 = np.dot(K, np.hstack((np.identity(3), np.zeros((3,1)))))
# P2_1 = np.dot(K, np.hstack((R1, t1.reshape((3,1)))))
# P2_2 = np.dot(K, np.hstack((R1, t2.reshape((3,1)))))

# proj1 = np.dot(P1, np.vstack((X, np.ones((1, n_pts)))))
# proj2_1 = np.dot(P2_1, np.vstack((X, np.ones((1, n_pts)))))
# proj2_2 = np.dot(P2_2, np.vstack((X, np.ones((1, n_pts)))))

# reproj_err1 = np.linalg.norm(proj1[:2,:] / proj1[2,:] - good_corners1.T, axis=0)
# reproj_err2_1 = np.linalg.norm(proj2_1[:2,:] / proj2_1[2,:] - good_corners2.T, axis=0)
# reproj_err2_2 = np.linalg.norm(proj2_2[:2,:] / proj2_2[2,:] - good_corners2.T, axis=0)

# print("Reprojection error (image 1): %.2f pixels" % np.mean(reproj_err1))
# print("Reprojection error (image 2, solution 1): %.2f pixels" % np.mean(reproj_err2_1))
# print("Reprojection error (image 2, solution 2): %.2f pixels" % np.mean(reproj_err2_2))

# import open3d as o3d
# import numpy as np

# # Load two images and convert to grayscale
# img1 = o3d.io.read_image("img1.png")
# img1 = np.asarray(img1)
# img1 = o3d.geometry.Image(img1.astype(np.uint8))

# img2 = o3d.io.read_image("img2.png")
# img2 = np.asarray(img2)
# img2 = o3d.geometry.Image(img2.astype(np.uint8))

# # Get the dimensions of the images
# dim1 = img1.get_max_bound()
# dim2 = img2.get_max_bound()

# # Create an Open3D pinhole camera object for each image
# cam1 = o3d.camera.PinholeCameraIntrinsic(dim1[0], dim1[1], fx=1000, fy=1000, cx=dim1[0]//2, cy=dim1[1]//2)
# cam2 = o3d.camera.PinholeCameraIntrinsic(dim2[0], dim2[1], fx=1000, fy=1000, cx=dim2[0]//2, cy=dim2[1]//2)

# # Compute depth maps from the two images
# depth1 = o3d.geometry.create_depth_from_pinhole(cam1, img1)
# depth2 = o3d.geometry.create_depth_from_pinhole(cam2, img2)

# # Create point clouds from the depth maps and camera intrinsics
# pcd1 = o3d.geometry.PointCloud.create_from_depth_image(depth1, cam1)
# pcd2 = o3d.geometry.PointCloud.create_from_depth_image(depth2, cam2)

# # Save the point clouds as PLY files
# o3d.io.write_point_cloud("pointcloud1.ply", pcd1, write_ascii=True)
# o3d.io.write_point_cloud("pointcloud2.ply", pcd2, write_ascii=True)

# # Load the two PLY files as Open3D point cloud objects
# pcd1 = o3d.io.read_point_cloud("pointcloud1.ply")
# pcd2 = o3d.io.read_point_cloud("pointcloud2.ply")

# # Visualize the two point clouds side by side
# o3d.visualization.draw_geometries([pcd1, pcd2])







import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load images
img_left = cv2.imread('img1.png')
img_right = cv2.imread('img2.png')

# Convert to grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Compute disparity map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(gray_left, gray_right)

# Compute 3D points
focal_length = 718.856
baseline = 0.573
Q = np.float32([[1, 0, 0, -img_left.shape[1]/2],
                [0, 1, 0, -img_left.shape[0]/2],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]])
points = cv2.reprojectImageTo3D(disparity, Q)

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot 3D points
ax.scatter(points[:,:,0], points[:,:,1], points[:,:,2], c=img_left.reshape(-1,3)/255.)

# Set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

# Show plot
plt.show()
