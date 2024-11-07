import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

IMAGE_SIZE = 32
BINARY_THRESHOLD = 0.49
SPLIT_SIZE = 16
K_NEIGHBORS = 1
SHOW_MISSTAKE = False


class DataReader():
    def get_images_from_assets():
        assets_path = "./assets/base/"
        images = []
        for file_name in os.listdir(assets_path):
            img=io.imread(assets_path+file_name, as_gray=True)   
            image = Image(img, int(file_name[0]))
            images.append(image)    
        return images

class Image():
    def __init__(self, img, number: int):
        self.img = img
        self.number = number
        self.vector = []

    def euclidean_distance(self, other):
        return np.linalg.norm(np.array(self.vector) - np.array(other.vector))

    def calculate_vector(self):
        sub = self.split_image()
        for s in sub:
            count_black = np.sum(s == 0)
            self.vector.append(count_black)
    
    def erosion(self):
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
        padded_image = np.pad(self.img, pad_width=1, mode='constant', constant_values=255)
        eroded_image = binary_erosion(padded_image, structure=structure).astype(self.img.dtype)
        eroded_image = eroded_image[1:-1, 1:-1]
        self.img = eroded_image
    
    def dilation(self):
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
        dilated_image = binary_dilation(self.img, structure=structure).astype(self.img.dtype)
        self.img = dilated_image
    
    def split_image(self):
        image = self.img
        height, width = image.shape
        sub_height = height // SPLIT_SIZE
        sub_width = width // SPLIT_SIZE
        sub_images = []
        for i in range(SPLIT_SIZE):
            for j in range(SPLIT_SIZE):
                start_y = i * sub_height
                start_x = j * sub_width
                end_y = start_y + sub_height
                end_x = start_x + sub_width
                sub_image = image[start_y:end_y, start_x:end_x]
                sub_images.append(sub_image)    
        return sub_images

    def grey_to_binary(self, threshold):
        binary_image = self.img < threshold
        self.img = binary_image
    
    def resize(self, size):
        x,y = self.img.shape    
        new_img = np.zeros((size, size))
        n = new_img.shape[0]
        m = new_img.shape[1]
        i_start = (size - x) // 2
        j_start = (size - y) // 2
        for i in range(n):
            for j in range(m):
                if i >= i_start and j >= j_start and i < x + i_start and j < y + j_start:
                    new_img[i][j] = self.img[i - i_start][j - j_start]
                else:
                    new_img[i][j] = 1
        self.img = new_img

def confusion_matrix(images):
    matrix = np.zeros((10, 10))
    distances = []
    for img in images:
        for img2 in images:
            if img2.vector != img.vector:
                distance = img.euclidean_distance(img2)
                distances.append((distance, img2))
        distances.sort(key=lambda x: x[0])        
        if K_NEIGHBORS == 1:
            choosen = distances[0][1]
            if choosen.number != img.number and SHOW_MISSTAKE:
                show_misclassified_images(img, choosen)
            choosen = choosen.number                
        else:
            choosen = k_nearest_neighbors(distances[:K_NEIGHBORS])
        # for i in range(K_NEIGHBORS):
        #     print(img.number, distances[i][1].number)   
        # print("choosen", choosen)     
        # print("----")
        matrix[img.number][choosen] += 1
        distances = []
    return matrix

def k_nearest_neighbors(distances):
    counts = np.zeros(10)
    for d in distances:
        counts[d[1].number] += 1    
    max_count = 0
    max_index = 0
    for i in range(10):
        if counts[i] > max_count:
            max_count = counts[i]
            max_index = i
    return max_index
    
def pourcentage(matrix):
    total = 0
    correct = 0
    for i in range(10):
        for j in range(10):
            total += matrix[i][j]
            if i == j:
                correct += matrix[i][j]
    return (correct / total) * 100

def show(image):
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.show()

def show_misclassified_images(img, chosen_img):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img.img, cmap='gray') 
    axes[0].set_title(f"Image testée: {img.number}")

    axes[1].imshow(chosen_img.img, cmap='gray') 
    axes[1].set_title(f"Image choisie: {chosen_img.number}")

    plt.show()

def show_specific_number(images, number): 
    matching_images = []
    for img in images:
        if img.number == number:
            matching_images.append(img.img)
    num_images = len(matching_images)    
    rows = int(np.ceil(num_images / 3))  
    cols = min(num_images, 3)     
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = np.array(axes).reshape(-1)
    for idx, img in enumerate(matching_images):
        ax = axes[idx]
        ax.imshow(img, cmap='gray', interpolation='nearest') 
    for ax in axes[num_images:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_all(images):
    for n in range(10):
        show_specific_number(images, n)
    
def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=plt.cm.Blues) 
    plt.title("Matrice de confusion", pad=20)
    fig.colorbar(cax)
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Classe réelle')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='black')
    plt.show()

def treat_images(images):
    for img in images:
        img.resize(IMAGE_SIZE)
        img.grey_to_binary(BINARY_THRESHOLD)
        img.dilation()        
        img.erosion()
        img.dilation()        
        img.erosion()        
        img.calculate_vector()    
    return images

if __name__ == "__main__":  
    images = DataReader.get_images_from_assets()    
    images = treat_images(images)
    matrix = confusion_matrix(images)
    plot_confusion_matrix(matrix)
    print(pourcentage(matrix))
    #show_specific_number(images, 9)