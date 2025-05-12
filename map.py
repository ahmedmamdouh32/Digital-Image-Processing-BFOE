import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import heapq

goal_icon = cv2.imread('location_icon.png',cv2.IMREAD_UNCHANGED)
goal_icon = cv2.cvtColor(goal_icon, cv2.COLOR_BGR2RGB)
goal_icon_percent = 10
width = int(goal_icon.shape[1] * goal_icon_percent / 100)
height = int(goal_icon.shape[0] * goal_icon_percent / 100)
dim = (width, height)
goal_icon = cv2.resize(goal_icon, dim, interpolation=cv2.INTER_AREA)

h, w = goal_icon.shape[:2]
print(w,h)
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        for neighbor in neighbors:
            x, y = neighbor
            if 0 <= x < rows and 0 <= y < cols:
                if grid[x][y] == 1:  # Walkable
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))

    return None  # No path found

#
# image = cv2.imread('supermarket_map.png')
# scale_percent = 20  # Resize to 20% of the original size
#
# # Calculate new dimensions
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
# # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("original map",image)

# cv2.imshow("B&W map",binary)

# height, width = binary.shape
#
# # نحسب عدد البكسلات
# binary = height * width


# grid = np.where(binary == 255, 1, 0)

# print(f"Number of pixels: {binary}")
# Load the image and create the binary grid as before
image = cv2.imread('supermarket_map2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

# Invert binary: 0 for path, 255 for obstacles (because dilation works on white)
binary_inverted = cv2.bitwise_not(binary)

# Inflate obstacles (dilate the obstacles)
kernel = np.ones((5,5), np.uint8)  # (5,5) kernel makes obstacles thicker
inflated = cv2.dilate(binary_inverted, kernel, iterations=1)
# Re-invert back: 255 for path, 0 for obstacle
inflated_binary = cv2.bitwise_not(inflated)
# cv2.imshow("we",inflated_binary)

# Now create the grid
grid = np.where(inflated_binary == 255, 1, 0)
print(image.shape)
start_x = 120
start_y = 400

start = (start_y, start_x)


goal_x = 620
goal_y = 550-1
goal = (goal_y, goal_x)




if grid[start[0]][start[1]] != 1:
    print("Start point is not walkable!")
if grid[goal[0]][goal[1]] != 1:
    print("Goal point is not walkable!")

path = astar(grid, start, goal)



if path:
    print("Path found!")

    for i, (x, y) in enumerate(path):
        if i % 5 == 0:  # يرسم كل 5 خطوات نقطة
            cv2.circle(image, (y,x), 1, (59, 191, 42), 2)

else:
    print("No path found.")



cv2.putText(image, "ME", (start_x-10, start_y-12), cv2.FONT_ITALIC, 0.5 , (0, 0, 0), 1)
# image[goal_y-50:goal_y+h-50, goal_x-25:goal_x+w-25] = goal_icon
cv2.circle(image, (start_x,start_y), 10, (255, 0, 0), -1)
cv2.circle(image, (goal_x,goal_y), 8, (7, 51, 227), -1)
# cv2.circle(inflated_binary, start, 2, (0, 255, 0), 2)
# cv2.circle(inflated_binary, goal, 2, (0, 255, 0), 2)
# cv2.imshow("we",inflated_binary)
cv2.imshow('Path', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
