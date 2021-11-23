import matplotlib
matplotlib.use('Agg')

# import package
import matplotlib.pyplot as plt

def save_plot(H, path):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H.history['loss'], label='train_loss')
    plt.plot(H.history['val_loss'], label='val_loss')
    plt.plot(H.history['accuracy'], label='train_acc')
    plt.plot(H.history['val_accuracy'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(path)