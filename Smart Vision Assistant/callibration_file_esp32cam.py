import cv2
import numpy as np
import glob
import os

# ==========================
# Chessboard Settings
# ==========================
CHECKERBOARD = (8, 6)       # Inner corners
SQUARE_SIZE = 0.025         # 25 mm

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================
# Prepare Object Points
# ==========================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

# ==========================
# Read Images
# ==========================
left_images = sorted(glob.glob("StereoImages/left/*.jpg"))
right_images = sorted(glob.glob("StereoImages/right/*.jpg"))

print("--------------------------------")
print("Stereo Calibration")
print("--------------------------------")

print("Left Images :", len(left_images))
print("Right Images:", len(right_images))

if len(left_images) == 0 or len(right_images) == 0:
    print("No images found.")
    exit()

valid_pairs = 0

# ==========================
# Detect Chessboard Corners
# ==========================

for left_file, right_file in zip(left_images, right_images):

    imgL = cv2.imread(left_file)
    imgR = cv2.imread(right_file)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(
        grayL,
        CHECKERBOARD,
        None
    )

    retR, cornersR = cv2.findChessboardCorners(
        grayR,
        CHECKERBOARD,
        None
    )

    if retL and retR:

        valid_pairs += 1

        objpoints.append(objp)

        cornersL = cv2.cornerSubPix(
            grayL,
            cornersL,
            (11,11),
            (-1,-1),
            criteria
        )

        cornersR = cv2.cornerSubPix(
            grayR,
            cornersR,
            (11,11),
            (-1,-1),
            criteria
        )

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        cv2.drawChessboardCorners(
            imgL,
            CHECKERBOARD,
            cornersL,
            retL
        )

        cv2.drawChessboardCorners(
            imgR,
            CHECKERBOARD,
            cornersR,
            retR
        )

        combined = np.hstack((imgL, imgR))

        cv2.imshow(
            "Detected Chessboard",
            combined
        )

        cv2.waitKey(100)

cv2.destroyAllWindows()

print("\nValid Image Pairs:", valid_pairs)

if valid_pairs < 15:
    print("Not enough valid images.")
    exit()

image_size = grayL.shape[::-1]

# ==========================================
# Left Camera Calibration
# ==========================================

print("\nCalibrating Left Camera...")

retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints,
    imgpointsL,
    image_size,
    None,
    None
)

print("Left Camera Calibration Done")

# ==========================================
# Right Camera Calibration
# ==========================================

print("\nCalibrating Right Camera...")

retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints,
    imgpointsR,
    image_size,
    None,
    None
)

print("Right Camera Calibration Done")

# ==========================================
# Stereo Calibration
# ==========================================

print("\nPerforming Stereo Calibration...")

flags = cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (
    cv2.TERM_CRITERIA_MAX_ITER +
    cv2.TERM_CRITERIA_EPS,
    100,
    1e-5
)

rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    K1,
    D1,
    K2,
    D2,
    image_size,
    criteria=criteria_stereo,
    flags=flags
)

print("\n====================================")
print("Stereo Calibration Finished")
print("====================================")

print("Stereo RMS Error :", rms)

print("\nLeft Camera Matrix")
print(K1)

print("\nRight Camera Matrix")
print(K2)

print("\nRotation Matrix")
print(R)

print("\nTranslation Vector")
print(T)



# ==========================================
# Stereo Rectification
# ==========================================

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    image_size,
    R,
    T,
    alpha=0
)

# ==========================================
# Save Calibration
# ==========================================

fs = cv2.FileStorage("stereo_calib2.xml", cv2.FILE_STORAGE_WRITE)

fs.write("K1", K1)
fs.write("D1", D1)

fs.write("K2", K2)
fs.write("D2", D2)

fs.write("R", R)
fs.write("T", T)

fs.write("R1", R1)
fs.write("R2", R2)

fs.write("P1", P1)
fs.write("P2", P2)

fs.write("Q", Q)

fs.release()

print("\n===================================")
print("stereo_calib2.xml saved successfully")
print("===================================")