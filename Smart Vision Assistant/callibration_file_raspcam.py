import cv2
import numpy as np
import glob
import os

# =====================================================
# CALIBRATION SETTINGS
# =====================================================

CHESSBOARD_SIZE = (6, 8)      # Inner corners (Columns, Rows)
SQUARE_SIZE = 0.025           # 25 mm = 0.025 meter

LEFT_FOLDER = "left_frames"
RIGHT_FOLDER = "right_frames"

# =====================================================
# TERMINATION CRITERIA
# =====================================================

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# =====================================================
# PREPARE OBJECT POINTS
# =====================================================

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# =====================================================
# STORAGE
# =====================================================

objpoints = []

imgpoints_left = []

imgpoints_right = []

# =====================================================
# LOAD IMAGE LIST
# =====================================================

left_images = sorted(glob.glob(os.path.join(LEFT_FOLDER, "*.png")))
right_images = sorted(glob.glob(os.path.join(RIGHT_FOLDER, "*.png")))

if len(left_images) == 0 or len(right_images) == 0:
    print("ERROR : Images not found.")
    exit()

if len(left_images) != len(right_images):
    print("ERROR : Left and Right image count mismatch.")
    exit()

print("------------------------------------")
print("Stereo Calibration")
print("------------------------------------")
print("Image Pairs :", len(left_images))
print()

valid_pairs = 0

# =====================================================
# PROCESS ALL IMAGE PAIRS
# =====================================================

for index, (left_name, right_name) in enumerate(zip(left_images, right_images), start=1):

    imgL = cv2.imread(left_name)
    imgR = cv2.imread(right_name)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE)

    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE)

    if retL and retR:

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

        objpoints.append(objp)

        imgpoints_left.append(cornersL)

        imgpoints_right.append(cornersR)

        valid_pairs += 1

        cv2.drawChessboardCorners(
            imgL,
            CHESSBOARD_SIZE,
            cornersL,
            retL
        )

        cv2.drawChessboardCorners(
            imgR,
            CHESSBOARD_SIZE,
            cornersR,
            retR
        )

        combined = cv2.hconcat([imgL, imgR])

        cv2.imshow("Detected Chessboard", combined)

        cv2.waitKey(200)

        print(f"[{index:02d}]  OK")

    else:

        print(f"[{index:02d}]  Chessboard NOT detected")

cv2.destroyAllWindows()

print()
print("------------------------------------")
print("VALID PAIRS :", valid_pairs)
print("------------------------------------")

if valid_pairs < 15:

    print("Not enough valid image pairs.")

    exit()

print("Stage 1 Completed Successfully.")

# =====================================================
# LEFT CAMERA CALIBRATION
# =====================================================

print("\n========================================")
print("Calibrating LEFT Camera...")
print("========================================")

retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints,
    imgpoints_left,
    grayL.shape[::-1],
    None,
    None
)

print("\nLEFT CAMERA RMS ERROR :", retL)

print("\nLEFT CAMERA MATRIX (K1)\n")
print(K1)

print("\nLEFT DISTORTION COEFFICIENTS (D1)\n")
print(D1)


# =====================================================
# RIGHT CAMERA CALIBRATION
# =====================================================

print("\n========================================")
print("Calibrating RIGHT Camera...")
print("========================================")

retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints,
    imgpoints_right,
    grayR.shape[::-1],
    None,
    None
)

print("\nRIGHT CAMERA RMS ERROR :", retR)

print("\nRIGHT CAMERA MATRIX (K2)\n")
print(K2)

print("\nRIGHT DISTORTION COEFFICIENTS (D2)\n")
print(D2)


# =====================================================
# FIX INTRINSIC PARAMETERS
# =====================================================

flags = cv2.CALIB_FIX_INTRINSIC


# =====================================================
# STEREO CALIBRATION
# =====================================================

print("\n========================================")
print("Stereo Calibration")
print("========================================")

criteria_stereo = (
    cv2.TERM_CRITERIA_MAX_ITER +
    cv2.TERM_CRITERIA_EPS,
    100,
    1e-5
)

stereo_error, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(

    objpoints,

    imgpoints_left,

    imgpoints_right,

    K1,

    D1,

    K2,

    D2,

    grayL.shape[::-1],

    criteria=criteria_stereo,

    flags=flags

)

print("\nStereo RMS Error :", stereo_error)

print("\nRotation Matrix (R)\n")
print(R)

print("\nTranslation Matrix (T)\n")
print(T)

baseline = np.linalg.norm(T)

print("\nBaseline :", baseline, "meters")

# =====================================================
# STEREO RECTIFICATION
# =====================================================

print("\n========================================")
print("Stereo Rectification")
print("========================================")

image_size = grayL.shape[::-1]

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

print("\nStereo Rectification Completed.")

print("\nR1\n")
print(R1)

print("\nR2\n")
print(R2)

print("\nP1\n")
print(P1)

print("\nP2\n")
print(P2)

print("\nQ Matrix\n")
print(Q)

# =====================================================
# SAVE CALIBRATION
# =====================================================

print("\n========================================")
print("Saving stereo_calib.xml")
print("========================================")

fs = cv2.FileStorage(
    "stereo_calib.xml",
    cv2.FILE_STORAGE_WRITE
)

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

print("\nCalibration File Saved Successfully!")

print("\nFile Name : stereo_calib.xml")

# =====================================================
# VERIFY XML
# =====================================================

print("\n========================================")
print("Verifying Calibration File")
print("========================================")

fs = cv2.FileStorage(
    "stereo_calib.xml",
    cv2.FILE_STORAGE_READ
)

Q_loaded = fs.getNode("Q").mat()

K1_loaded = fs.getNode("K1").mat()

K2_loaded = fs.getNode("K2").mat()

fs.release()

print("K1 Shape :", K1_loaded.shape)
print("K2 Shape :", K2_loaded.shape)
print("Q Shape :", Q_loaded.shape)

print("\nCalibration Verified Successfully.")