import matplotlib.pyplot as plt

def plot_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.show()