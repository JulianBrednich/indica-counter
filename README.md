# Indiaca Counter

A small learning project: counting rally passes in Indiaca videos with plain OpenCV, without machine learning.

I built it to get hands-on with classic computer vision in Python: colour segmentation, simple tracking and making the result hold up on real footage.

## What it does

The script follows the red shuttle through a video filmed from the side, with the net in the middle of the frame. It counts one pass every time the shuttle crosses the centre line.

The detection itself is just an HSV colour threshold. Most of the work went into the rules around it:

- **Ignore static red objects.** A blob that was already in the same place in one of the last few frames is treated as part of the scene.
- **Follow the shuttle.** If several moving blobs show up, the one closest to the last known position is used, up to a maximum jump distance.
- **Check the flight direction.** A position that would mean a sharp change of direction (more than 45°) between frames is rejected as a false detection.
- **Re-acquire carefully.** After the shuttle has been lost for a while, it has to be seen in a few frames in a row before it counts again.
- **Count only plausible crossings.** A crossing only counts if the horizontal movement over the last frames is in a realistic range.

Pixel thresholds are defined for 1080p and scaled to the video resolution.

## Results

I tested it on five of my own videos. The count was correct in all of them; the longest had 112 passes. These were shot under similar conditions, so this is not a general benchmark.

## Usage

```bash
pip install -r requirements.txt

python indiaca_counter.py my_video.mov               # show the video with tracking overlay
python indiaca_counter.py my_video.mov --tune        # adjust the HSV thresholds with sliders
python indiaca_counter.py my_video.mov --no-display  # only print the final count
```

While the video is shown: `Space` pauses, `Esc` quits.

Overlay colours: green boxes are moving candidates, red boxes are ignored static objects, the filled blue square is the tracked shuttle.

## Limitations

- Colour-based detection depends on lighting and background. Other red objects that move, or a different shuttle colour, need re-tuning (`--tune`).
- Assumes a fixed camera with the net roughly in the centre of the image.
- The thresholds were tuned by hand on my own footage.
