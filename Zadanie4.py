import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def count_lakes(image):
    b = ~image
    frame = np.ones((b.shape[0]+2, b.shape[1]+2))
    frame[1:-1,1:-1] = b
    return np.max(label(frame))-1
    
def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
    return 1 in lines

def has_gline(image):
    lines = np.sum(image, 1) // image.shape[1]
    return 1 in lines

def count_bays(image):
    b = ~image
    return np.max(label(b)) - count_lakes(image)

def recognize(region):
    lakes = count_lakes(region.image)
    if lakes == 2:
        if has_vline(region.image):
            return "B"
        return "8"
    elif lakes == 1:
        bays = count_bays(region.image)
        if bays == 4:
            return "0"
        if bays == 3:
            return "A"
        if bays == 2:
            if region.eccentricity < 0.60:
                return "D"
            else:
                return "P"
    elif lakes == 0:
        if has_vline(region.image):   
            if np.all(region.image == 1):
                return "-"
            return "1"
        bays = count_bays(region.image)
        if bays == 2:
            return "/"
        circ = region.perimeter ** 2 / region.area
        if circ > 70:
            return "*"
        if bays == 4:
            return "X"
        if bays == 5:
            return "W"
    else:
        print("something")
    return "None"


image = plt.imread("symbols.png")
if image.ndim == 4:
    image = image[:,:,:-1]
    
binary = np.sum(image,2)
binary[binary > 0] = 1

labeled = label(binary)
regions = regionprops(labeled)

d = {"None":0}

for region in regions:
    symbol = recognize(region)
    if symbol not in d:
        d[symbol] = 1
    else:
        d[symbol] += 1
print("\nЧастотный словарь:\n")
print(f"Общее количество символов: {np.max(labeled)}")
for key in sorted(d):
    if not key == "None":
        print(f'Символ "{key}": {d[key]}')
  
percent = 1 - d["None"]/np.max(labeled)    
print(f"\nПроцент распознавания символов: {percent*100}%")

