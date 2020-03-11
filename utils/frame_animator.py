import imageio
from IPython.display import HTML
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt


class FrameAnimator(object):
    def __init__(self, imgs, frame_start_i=0, frame_stop_i=None, interval=100, blit=True, text=None, figsize=(6, 6)):
        if frame_stop_i is None:
            frame_stop_i = len(imgs)

        if len(imgs[0].shape) == 2 or imgs[0].shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        fig, ax = plt.subplots(1, figsize=figsize)

        img_0 = imgs[0]
        if img_0.shape[-1] == 1:
            img_0 = np.repeat(img_0, 3, axis=2)

        video_img = ax.imshow(img_0, cmap=cmap)
        ax.axis('off')

        def init():
            img_0 = imgs[0]
            if img_0.shape[-1] == 1:
                img_0 = np.repeat(img_0, 3, axis=2)
            video_img.set_array(np.zeros_like(img_0))
            return (video_img,)

        def animate(i):
            img_i = imgs[i]
            if img_i.shape[-1] == 1:
                img_i = np.repeat(img_i, 3, axis=2)

            video_img.set_array(img_i)

            text_str = f' -- {text[i]}' if text is not None else ""
            ax.set_title(f"Frame {frame_start_i+i+1}/{frame_stop_i} {text_str}")

            return (video_img,)

        plt.tight_layout()
        plt.close()  # prevent fig from showing
        self.anim = animation.FuncAnimation(fig, animate, init_func=init,
                                            frames=len(imgs), interval=interval,
                                            # delay between frames in ms (25FPS=25 f/s * 1 s/1000ms = 0.025 f/ms)
                                            blit=blit)

    def __call__(self, mode='jsHTML'):
        if mode == 'HTML':
            return HTML(self.anim.to_html5_video())
        elif mode == 'jsHTML':
            return HTML(self.anim.to_jshtml())


def read_gif(path, figsize=(8,8)):
    images = imageio.mimread(path)
    animator = FrameAnimator(images, figsize=figsize)
    return animator()