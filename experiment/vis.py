import os
from flask import Flask, render_template_string, send_from_directory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

app = Flask(__name__)

# Change this path to the directory containing your videos
VIDEO_DIR = args.dir
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Viewer</title>
</head>
<body>
    {% for video in videos %}
        <div>
            <p>{{ loop.index }}. {{ video }}</p>
            <video width="640" controls>
                <source src="{{ url_for('serve_video', filename=video) }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    {% endfor %}
</body>
</html>
"""

@app.route('/')
def index():
    if not os.path.exists(VIDEO_DIR):
        return f"Directory {VIDEO_DIR} does not exist."
    
    videos = []
    for f in sorted(os.listdir(VIDEO_DIR)):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            videos.append(f)
            
    return render_template_string(HTML_TEMPLATE, videos=videos)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
