import re

def open_ffmpeg_stream_adaptive(url):
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-an",
        "pipe:1"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read stderr line by line until we find the resolution
    width, height = None, None
    while True:
        line = process.stderr.readline().decode("utf-8", errors="ignore")
        if not line:
            break
        match = re.search(r"(\d{2,5})x(\d{2,5})", line)
        if match and "Video:" in line:
            width, height = int(match.group(1)), int(match.group(2))
            print(f"Detected resolution: {width}x{height}")
            break

    if not width or not height:
        process.terminate()
        raise RuntimeError("Could not detect stream resolution from FFmpeg output.")

    # Drain stderr in a background thread to prevent blocking
    import threading
    threading.Thread(target=lambda: process.stderr.read(), daemon=True).start()

    return process, width, height