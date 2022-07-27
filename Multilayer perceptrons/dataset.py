import matplotlib.pyplot as plt
import torch
import torchvision

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# print(get_fashion_mnist_labels([mnist_train[i][1] for i in range(len(mnist_train))])[0:10])

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, (ax, img) in enumerate(zip(axes.flat, imgs)):
        if torch.is_tensor(img):
            # Tensor image
            ax.imshow(img.numpy())
        else:
            # PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

# iterator = iter(torch.utils.data.DataLoader(mnist_train, batch_size=18))
# for i in range(5):
#     X, y = next(iterator)
#     show_images(X.reshape(18, 28, 28), 2, 9, titles = get_fashion_mnist_labels(y))

def load_data_fashion_mnist(batch_size, resize=None):
    if not resize:
        transform = torchvision.transforms.ToTensor() # Converts image from PIL to 32-bit floating point tensors (all pixel values bvetween 0 and 1)
    else:
        transform = [torchvision.transforms.ToTensor()]
        transform.insert(0, torchvision.transforms.Resize(resize))
        transform = torchvision.transforms.Compose(transform)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=transform, download=True)

    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4))

if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(32, 64)
    for i in range(2):
        X, y = next(iter(train_iter))
        show_images(X.reshape(32, 64, 64), 4, 8, titles = get_fashion_mnist_labels(y))
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
