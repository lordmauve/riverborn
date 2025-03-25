"""Video recording and screenshot functions.

Adapted from Wasabi2D, https://github.com/lordmauve/wasabi2d
"""
import moderngl_window as mglw
import moderngl
import typing
import imageio as iio
import numpy as np


if typing.TYPE_CHECKING:
    import subprocess


def capture_screen(fb: moderngl.Framebuffer) -> np.ndarray:
    """Capture contents of the given framebuffer as a Pygame Surface."""
    width = fb.width
    height = fb.height
    data = fb.read(components=3)
    assert len(data) == (width * height * 3), \
        f"Received {len(data)}, expected {width * height * 3}"
    img_data = np.frombuffer(data, dtype='u1', count=width * height * 3)
    img_data = img_data.reshape((height, width, 3))
    return np.flipud(img_data)  # Flip the image vertically


def screenshot(filename: str | None = None) -> str:
    """Take a screenshot.

    If filename is not given, save to a file named screenshot_<time>.png
    in the current directory. Return the filename.

    """
    import datetime
    ctx = mglw.ctx()
    if filename is None:
        now = datetime.datetime.now()
        filename = f'screenshot_{now:%Y-%m-%d_%H:%M:%S.%f}.png'
    img_data = capture_screen(ctx.screen)
    iio.imwrite(filename, img_data)
    print(f"Wrote screenshot to {filename}")
    return filename


class VideoRecorder:
    ctx: moderngl.Context
    _recording: str | None
    _ffmpeg: 'subprocess.Popen | None'
    _real_size: tuple[int, int]

    def __init__(self):
        self.ctx = mglw.ctx()
        self._recording = None
        self._ffmpeg = None
        self._real_size = self.ctx.screen.size

    def record_video(self, filename=None):
        """Start recording a video.

        Video will be encoded in MPEG4 format.

        This requires an ffmpeg binary to be located on $PATH.

        If filename is not given, save to a file named video_<time>.mp4
        in the current directory.
        """
        import subprocess
        import datetime
        if not filename:
            now = datetime.datetime.now()
            filename = f'video_{now:%Y-%m-%d_%H:%M:%S.%f}.mp4'
        self._recording = filename
        w, h = self._real_size
        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',  # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '60',  # frames per second
            '-i', '-',  # The imput comes from a pipe
            '-vf', 'vflip',

            # These options are needed for uploads to Twitter
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-strict', '-2',
            '-an',  # Tells FFMPEG not to expect any audio
            filename,
        ]
        self._ffmpeg = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            bufsize=0
        )
        print("Recording video...")

    def stop_recording(self):
        """Finish recording the current video."""
        self._ffmpeg.stdin.close()
        ret = self._ffmpeg.wait()
        if ret == 0:
            print("Saved recording to", self._recording)
        else:
            print("Error writing video.")
        self._recording = None

    def toggle_recording(self) -> bool:
        """Start or stop recording video.

        Return True if recording started.
        """
        if not self._recording:
            self.record_video()
            return True
        else:
            self.stop_recording()

    def _vid_frame(self):
        if self._recording:
            data = self.ctx.screen.read(components=3)
            self._ffmpeg.stdin.write(data)
