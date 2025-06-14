import subprocess
import os

class CombineVideo:
    def __init__(self, video1_path, video2_path, output_path, stack_direction='horizontal'):
        """
        Initialize video combination settings.

        Args:
            video1_path (str): Path to the first input video.
            video2_path (str): Path to the second input video.
            output_path (str): Output path for the combined video.
            stack_direction (str): 'horizontal' or 'vertical' (default is 'horizontal').
        """
        self.video1 = video1_path
        self.video2 = video2_path
        self.output = output_path
        self.stack_direction = stack_direction

    def stack_videos(self):
        """
        Run FFmpeg to stack videos.
        """
        if self.stack_direction not in ['horizontal', 'vertical']:
            raise ValueError("stack_direction must be 'horizontal' or 'vertical'")

        filter_type = 'hstack=inputs=2' if self.stack_direction == 'horizontal' else 'vstack=inputs=2'

        ffmpeg_command = [
            "ffmpeg", "-i", self.video1, "-i", self.video2, "-filter_complex", f"[0:v][1:v]{filter_type}",
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", self.output]

        subprocess.run(ffmpeg_command, check=True)
        print(f"Video stacking complete: {self.output}")
