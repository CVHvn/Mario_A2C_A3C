from moviepy.editor import VideoFileClip
from moviepy.editor import *
import glob

def mp4_to_gif(mp4_path, gif_path, max_duration=60):
    # Load the video file
    clip = VideoFileClip(mp4_path)
    
    # Calculate speedup factor
    speedup_factor = clip.duration / max_duration
    
    # If the video duration is within the limit, no need for speedup
    if clip.duration <= max_duration:
        clip.write_gif(gif_path, fps=10)
    else:
        speedup_clip = clip.fx( vfx.speedx, 2) 
        speedup_clip.write_gif(gif_path, fps=clip.fps / speedup_factor)

# Example usage

files = glob.glob("*.mp4")
for f in files:
  file_name = f.split(".")[0]
  print(f, file_name, f"{file_name}.gif")
  mp4_to_gif(f, f"{file_name}.gif")