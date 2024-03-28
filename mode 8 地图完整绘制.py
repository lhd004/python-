import cv2
import numpy as np


maze_matrix =[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

CELL_SIZE = 30

def draw_maze(maze_matrix):
    maze_height = len(maze_matrix)
    maze_width = len(maze_matrix[0])
    maze_image = np.zeros((maze_height * CELL_SIZE, maze_width * CELL_SIZE, 3), dtype=np.uint8)

    for i in range(maze_height):
        for j in range(maze_width):
            if maze_matrix[i][j] == 0:
                color = (0, 0, 0)  # 墙的颜色为黑色
            elif maze_matrix[i][j] == 2:
                color = (0, 255, 0)  # 值为2时，颜色为绿色
            elif maze_matrix[i][j] == 3:
                color = (255, 0, 0)  # 值为3时，颜色为蓝色
            elif maze_matrix[i][j] == 4:
                color = (0, 0, 255)  # 值为4时，颜色为红色
            else:
                color = (255, 255, 255)  # 路的颜色为白色

            cv2.rectangle(maze_image, (j * CELL_SIZE, i * CELL_SIZE), ((j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE), color, -1)

            if (i, j) in where_treasure:
                center = ((j * CELL_SIZE + CELL_SIZE // 2), (i * CELL_SIZE + CELL_SIZE // 2))  # 计算单元格中心点坐标
                cv2.circle(maze_image, center, CELL_SIZE // 4, (0, 0, 0), -1)  # 绘制黑色圆，半径设为单元格大小的四分之一
    return maze_image

def detect_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_count = 0
    square_centers = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if 500 < area < 2000:
                # Calculate center of the square
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    square_centers.append((cX, cY))
                square_count += 1

    if square_count == 4:
        return True, square_centers

    return False, None



cap = cv2.VideoCapture(0)
detected = False
square_centers = None

while not detected:
    ret, frame = cap.read()
    if not ret:
        break

    detected, square_centers = detect_squares(frame)

    cv2.imshow('Square Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

if detected:
    # Order points and warp perspective
    rect = np.array(square_centers, dtype="float32")
    (bl, br, tl, tr) = rect  # Reorder points according to new requirement

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
        [0, 0],
        [maxWidth - 1, 0]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    # Crop image by 40 pixels on each side
    cropped = warped[40:maxHeight-40, 40:maxWidth-40]

    # Convert cropped image to grayscale
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cropped_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=20, minRadius=5, maxRadius=30)

    # Initialize 10x10 array to store circle detection results
    #grid = np.zeros((10, 10), dtype=int)
    treasure = np.zeros((21, 21), dtype=int)
    where_treasure = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Calculate grid index
            row = int((y / cropped.shape[0]) * 10)
            col = int((x / cropped.shape[1]) * 10)
            # Mark the corresponding grid cell as detected
            #grid[row][col] = 1
            #treasure[row * 2 + 1][col * 2 + 1] = 1
            where_treasure.append((row * 2 + 1, col * 2 + 1))
            #if len(where_treasure) == 8:
                #where_treasure.append((1,19))
    #map1 = np.array(maze_matrix)
    #map2 = np.array(treasure)
    #map3 = map1 + map2
    print("Grid with circle detection results:")
    #print(grid)
    #print(treasure)
    print(where_treasure)
    maze_image = draw_maze(maze_matrix)
    cv2.imshow('Maze', maze_image)


cv2.waitKey(0)
cv2.destroyAllWindows()