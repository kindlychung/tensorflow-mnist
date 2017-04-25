import matplotlib.pyplot as plt

def show_flat_image(flat, width, height, cmap="gray"):
    plt.close()
    plt.imshow(flat.reshape([width, height]), cmap=cmap)
    plt.show(block=False)