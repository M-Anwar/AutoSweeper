import imageio
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import distance as dist
import pyautogui
import time
import os
pyautogui.PAUSE = 0.1

DIFFICULTIES = ['medium']
CELL_DEF = {
    "medium": (14,18)
}

# Background Color Definitions
BACKGROUND_COLORS = [(229,193,161), (215,183,155), (136,174,70), (171,213,94),  (163, 207, 86)]
UNCLICKABLE = [(229,193,161), (215,183,155)]
CLICKABLE = [(171,213,94),  (163, 207, 86)]

# Cell grid number color definitions
NUMBER_COLORS = [(27,121,206), (63,142,69), (210,51,54), (134,54,158), (254,146,0), (14,152,166)]

class Cell(object):
    def __init__(self, value, left, top, width,height):
        self.value = value
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self.mouse_center = (left+width/2, top+height/2)

class SweeperGrid(object):
    def __init__(self, difficulty='medium'):
        if difficulty not in DIFFICULTIES:
            raise Exception("Only {} difficulties supported. You passed: {}".format(DIFFICULTIES, difficulty))
        
        medium_grid = cv2.imread('resources/medium_grid.png', cv2.IMREAD_GRAYSCALE) 
        
        # Visualization Params
        self.screen_vis = None

        # Locate the grid and save where it is
        (self.x_min, self.y_min),(self.x_max, self.y_max) = self.getGridPosition(medium_grid)
        self.grid_w, self.grid_h = (self.x_max-self.x_min, self.y_max-self.y_min)
        
        # Compute and initialze each grid cell
        self.rows, self.cols = CELL_DEF[difficulty]
        self.cell_w, self.cell_h = (self.grid_w/self.cols, self.grid_h/self.rows)
        x_grid = np.linspace(0, self.grid_w, num = self.cols, endpoint=False)
        y_grid = np.linspace(0, self.grid_h, num = self.rows, endpoint=False)
        self.cells = [[Cell(-1, self.x_min+x, self.y_min+y, self.cell_w, self.cell_h) for x in x_grid] for y in y_grid]


    def getGridPosition(self, grid_template):
        screen = np.asarray(imageio.imread('<screen>'))
        self.screen_vis = screen
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)          
        (tH, tW) = grid_template.shape[:2]

        found = None        
        for scale in np.linspace(0.2, 1.0, 100)[::-1]:        
            resized = np.array(Image.fromarray(screen).resize( (int(screen.shape[1] * scale), int(screen.shape[0] * scale)) ))      
            r = screen.shape[0] / float(resized.shape[0])        
            
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break        
            
            result = cv2.matchTemplate(resized, grid_template, cv2.TM_CCOEFF_NORMED)
            
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)       
            
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)       
                if maxVal > 0.99:
                    break
            
        (maxVal, maxLoc, r) = found
        if maxVal < 0.9:
            raise Exception("Unable to find a suitable playing grid")
            
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))    
        
        return (startX, startY), (endX, endY)

    def updateGrid(self):
        screen = np.asarray(imageio.imread('<screen>'))
        self.screen_vis = screen
        cells = []
        for row in self.cells:
            for col in row:
                cells.append(screen[col.top:col.top+col.height, col.left: col.left+col.width].copy())
        cells = np.stack(cells)
        cell_masks = self._getMaskedCells(cells, BACKGROUND_COLORS)

        for i, (cell, mask) in enumerate(zip(cells, cell_masks)):
            row, col = (int(i/self.cols), int(i%self.cols))           
            mean = cv2.mean(cell, mask = mask.astype('uint8'))[:3]
            if np.sum(mean) > 50:
                minDist = self._getClosestColor(mean, NUMBER_COLORS)
                self.cells[row][col].value = minDist[1] + 1
            else:
                mean = cv2.mean(cell)[:3]        
                minDist = self._getClosestColor(mean, CLICKABLE + UNCLICKABLE)            
                if minDist[1]<=1:  
                    self.cells[row][col].value = -1                            
                else:
                    self.cells[row][col].value = -2 
    
    def updateMines(self, mine_locations):
        for mines in mine_locations:           
            self.cells[mines[0]][mines[1]].value -3

    def _getMaskedCells(self, cells, background_colors):
        final_mask = np.zeros(cells.shape[:-1])   
        for color in background_colors:        
            final_mask = np.logical_or(np.all(cells == color, axis=-1), final_mask)
        border = np.ones(final_mask.shape).astype('bool')
        border[:,5:-5,5:-5] = False
        final_mask = np.logical_or(final_mask, border)
        return np.invert(final_mask)

    def _getClosestColor(self, mean, colors):
        minDist = (np.inf, None)        
        for i, col in enumerate(colors):
            d = dist.euclidean(np.array(col), mean)
            if d < minDist[0]:
                minDist = (d,i) 
                
        return minDist


    def toArray(self):
        """
        Returns a numpy array representation of the grid
        """
        arr = np.zeros((self.rows, self.cols))
        for i,row in enumerate(self.cells):
            for j,col in enumerate(row):
                arr[i,j] = col.value
        return arr

    def visualizeGrid(self):
        visualization = self.screen_vis.copy()
        cv2.rectangle(visualization, (self.x_min, self.y_min), (self.x_max, self.y_max), (255,0,0), 2)

        for row in self.cells:
            for cell in row:
                cv2.rectangle(visualization, (cell.left, cell.top), (cell.left+cell.width, cell.top+cell.height), (0,0,255), 1)
                cv2.putText(visualization, '{}'.format(cell.value), (int(cell.left+cell.width/3), int(cell.top+cell.height/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return Image.fromarray(visualization)

    def __getitem__(self, index):
        return self.cells[index]

    def saveGridImage(self, path):
        screen = np.asarray(imageio.imread('<screen>'))[self.y_min:self.y_max, self.x_min:self.x_max]
        Image.fromarray(screen).save(path)


def getEasyPlaces(board):
    known_mines = set()
    known_safe = set()

    for edges in np.argwhere(board>0):
        row,col = edges[0],edges[1]
        
        total_open = []
        for dx in range(-1,2):
            for dy in range(-1,2):
                drow, dcol = row+dy, col+dx
                if drow<0 or drow>=board.shape[0] or dcol<0 or dcol>=board.shape[1]:
                    continue
                if board[drow,dcol] == -1:
                    total_open.append((drow,dcol))                  
        
        if len(total_open) == board[row,col]:        
            known_mines.update(total_open[0:int(board[row,col])])

    for mines in known_mines:
        board[mines[0],mines[1]] = -3
        
    for edges in np.argwhere(board>0):
        row,col = edges[0],edges[1]
        
        total_open = []
        total_mines = 0
        for dx in range(-1,2):
            for dy in range(-1,2):
                drow, dcol = row+dy, col+dx
                if drow<0 or drow>=board.shape[0] or dcol<0 or dcol>=board.shape[1]:
                    continue
                if board[drow,dcol] == -1:
                    total_open.append((drow,dcol))
                elif board[drow,dcol] == -3:
                    total_mines +=1                
                    
        if total_mines == board[row,col]:
            known_safe.update(total_open)

    return known_mines, known_safe

if __name__ == "__main__":    
    os.makedirs("playback", exist_ok=True)

    sweeper = SweeperGrid("medium")    

    print("Making First Move")
    row,col = np.random.randint(0,sweeper.rows), np.random.randint(0,sweeper.cols)
    first_pos = sweeper[row][col].mouse_center
    pyautogui.moveTo(x=first_pos[0], y=first_pos[1], duration=0.1)
    pyautogui.click()
    pyautogui.click()
    time.sleep(1)
    sweeper.updateGrid()

    count=0
    while True:        
        print("Making Move", count)
        sweeper.visualizeGrid().save("playback/move_{}.png".format(count))      
        board = sweeper.toArray()
        known_mines, known_safe = getEasyPlaces(board)
        sweeper.updateMines(known_mines)

        if len(known_safe) == 0:
            break

        for safe in known_safe:
            pos = sweeper[safe[0]][safe[1]].mouse_center
            pyautogui.moveTo(x=pos[0], y=pos[1], duration=0.1)
            pyautogui.click()

        pyautogui.moveTo(10,10)
        time.sleep(1)
        sweeper.updateGrid()         
        count+=1       
       
    print("Finished playing, no more moves to make")