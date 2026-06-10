import os
from flask import Flask, render_template_string, send_from_directory
import argparse

parser = argparse.ArgumentParser(description="Visualize video files in a directory and its subdirectories.")
parser.add_argument('--dir', type=str, required=True, help="Directory containing videos")
parser.add_argument('--port', type=str, default=5000, help="Directory containing videos")
args = parser.parse_args()

app = Flask(__name__)

VIDEO_DIR = os.path.abspath(args.dir)

ERROR_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Video Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #090d16;
            color: #f3f4f6;
            font-family: 'Outfit', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .error-card {
            background: #111827;
            border: 1px solid rgba(239, 68, 68, 0.25);
            padding: 40px;
            border-radius: 16px;
            max-width: 500px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }
        h1 {
            color: #f87171;
            margin-top: 0;
            font-size: 1.8rem;
            letter-spacing: -0.5px;
        }
        p {
            color: #9ca3af;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .path {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 10px 14px;
            border-radius: 8px;
            font-family: monospace;
            word-break: break-all;
            display: inline-block;
            margin: 15px 0;
            color: #a5b4fc;
        }
    </style>
</head>
<body>
    <div class="error-card">
        <h1>Directory Not Found</h1>
        <p>The specified video directory does not exist or cannot be accessed:</p>
        <div class="path">{{ dir }}</div>
        <p>Please check your --dir parameter and try again.</p>
    </div>
</body>
</html>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Explorer - {{ dir_name }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: #111827;
            --bg-sidebar: #0b0f19;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --border-color: rgba(255, 255, 255, 0.07);
            --border-hover: rgba(99, 102, 241, 0.4);
            --shadow-main: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            --border-radius: 12px;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .app-container {
            display: flex;
            width: 100%;
            min-height: 100vh;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 290px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            flex-shrink: 0;
            height: 100vh;
            position: sticky;
            top: 0;
            box-sizing: border-box;
            z-index: 10;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sidebar-logo {
            background: var(--accent-gradient);
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
        }

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.5px;
            color: var(--text-primary);
            margin: 0;
        }

        .sidebar-subtitle {
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            word-break: break-all;
            margin-top: 4px;
        }

        .folder-list-container {
            flex-grow: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding-right: 4px;
        }

        .folder-list-container::-webkit-scrollbar {
            width: 5px;
        }
        .folder-list-container::-webkit-scrollbar-track {
            background: transparent;
        }
        .folder-list-container::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        .folder-item {
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-align: left;
            user-select: none;
        }

        .folder-item:hover {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-primary);
        }

        .folder-item.active {
            background: var(--accent-gradient);
            color: white;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
            border-color: rgba(255, 255, 255, 0.05);
        }

        .folder-item-left {
            display: flex;
            align-items: center;
            gap: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .folder-badge {
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 20px;
            font-size: 0.75rem;
            flex-shrink: 0;
        }

        .folder-item.active .folder-badge {
            background: rgba(0, 0, 0, 0.2);
        }

        /* Main Content Styling */
        .main-content {
            flex-grow: 1;
            padding: 40px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 30px;
            overflow-y: auto;
            height: 100vh;
        }

        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }

        .page-title-section h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0;
            background: linear-gradient(to right, #ffffff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .page-title-section p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin: 4px 0 0 0;
        }

        .search-container {
            position: relative;
            width: 320px;
        }

        .search-input {
            width: 100%;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 12px 16px 12px 42px;
            border-radius: var(--border-radius);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        .search-icon {
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            pointer-events: none;
            display: flex;
            align-items: center;
        }

        /* Video Grid & Cards */
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
        }

        .video-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-main);
        }

        .video-card:hover {
            transform: translateY(-4px);
            border-color: var(--border-hover);
            box-shadow: 0 12px 25px -10px rgba(99, 102, 241, 0.3);
        }

        .video-container {
            position: relative;
            background: #000;
            width: 100%;
            aspect-ratio: 16/10;
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid var(--border-color);
        }

        .video-player {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .video-info {
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            flex-grow: 1;
        }

        .video-title-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .video-title {
            font-weight: 500;
            font-size: 0.95rem;
            line-height: 1.4;
            color: var(--text-primary);
            word-break: break-all;
            margin: 0;
        }

        .video-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: auto;
            padding-top: 8px;
        }

        .path-badge {
            background: rgba(99, 102, 241, 0.1);
            color: #818cf8;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .video-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }

        .action-btn {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s ease;
        }

        .action-btn:hover {
            background: rgba(99, 102, 241, 0.1);
            color: #818cf8;
            border-color: rgba(99, 102, 241, 0.3);
        }

        .action-btn svg {
            flex-shrink: 0;
        }

        /* Empty State */
        .no-videos {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 80px 20px;
            text-align: center;
            background: var(--bg-card);
            border: 1px dashed var(--border-color);
            border-radius: var(--border-radius);
            color: var(--text-secondary);
        }

        .no-videos svg {
            margin-bottom: 16px;
            color: rgba(99, 102, 241, 0.4);
        }

        .no-videos h3 {
            margin: 0 0 8px 0;
            color: var(--text-primary);
            font-size: 1.2rem;
        }

        .no-videos p {
            margin: 0;
            font-size: 0.95rem;
            max-width: 400px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-logo">V</div>
                <div>
                    <h2 class="sidebar-title">Video Explorer</h2>
                    <div class="sidebar-subtitle" title="{{ dir_path }}">{{ dir_name }}</div>
                </div>
            </div>

            <div class="folder-list-container">
                <div class="folder-item active" data-folder="all" onclick="setFolder('all')">
                    <div class="folder-item-left">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
                        <span>All Folders</span>
                    </div>
                    <span class="folder-badge">{{ videos|length }}</span>
                </div>
                {% for folder in folders %}
                <div class="folder-item" data-folder="{{ folder }}" onclick="setFolder('{{ folder }}')">
                    <div class="folder-item-left">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
                        <span title="{{ folder }}">{{ folder }}</span>
                    </div>
                    <span class="folder-badge">{{ folder_counts[folder] }}</span>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <div class="top-bar">
                <div class="page-title-section">
                    <h1>Videos (<span id="video-count">{{ videos|length }}</span>)</h1>
                    <p>Click items to filter or copy paths for commands</p>
                </div>
                <div class="search-container">
                    <div class="search-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    </div>
                    <input type="text" class="search-input" placeholder="Search videos..." oninput="handleSearch(this)">
                </div>
            </div>

            <!-- Videos Grid -->
            <div class="video-grid">
                {% for video in videos %}
                <div class="video-card" data-folder="{{ video.folder }}" data-name="{{ video.filename }}">
                    <div class="video-container">
                        <video class="video-player" controls preload="metadata">
                            <source src="{{ url_for('serve_video', filename=video.rel_path) }}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <div class="video-info">
                        <div class="video-title-container">
                            <h3 class="video-title" title="{{ video.filename }}">{{ video.filename }}</h3>
                        </div>
                        <div class="video-meta">
                            <div class="path-badge" title="Sub-folder: {{ video.folder }}">
                                📁 {{ video.folder }}
                            </div>
                        </div>
                        <div class="video-actions">
                            <button class="action-btn" onclick="copyToClipboard('{{ video.rel_path }}', this)" title="Copy relative path for scripts/cat.py">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                <span>Copy Path</span>
                            </button>
                            <a href="{{ url_for('serve_video', filename=video.rel_path) }}" target="_blank" class="action-btn" title="Open video in new tab">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                                <span>Open</span>
                            </a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Empty State -->
            <div id="no-videos" class="no-videos">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg>
                <h3>No Videos Found</h3>
                <p>No video files match your current search query or folder selection.</p>
            </div>
        </div>
    </div>

    <script>
        let currentFolder = 'all';
        let searchQuery = '';

        function setFolder(folderName) {
            currentFolder = folderName;
            document.querySelectorAll('.folder-item').forEach(item => {
                if (item.getAttribute('data-folder') === folderName) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
            filterVideos();
        }

        function handleSearch(inputEl) {
            searchQuery = inputEl.value.toLowerCase();
            filterVideos();
        }

        function filterVideos() {
            const cards = document.querySelectorAll('.video-card');
            let visibleCount = 0;

            cards.forEach(card => {
                const cardFolder = card.getAttribute('data-folder');
                const cardName = card.getAttribute('data-name').toLowerCase();

                const matchesFolder = (currentFolder === 'all' || cardFolder === currentFolder);
                const matchesSearch = cardName.includes(searchQuery);

                if (matchesFolder && matchesSearch) {
                    card.style.display = 'flex';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });

            document.getElementById('video-count').textContent = visibleCount;

            const noVideosEl = document.getElementById('no-videos');
            if (visibleCount === 0) {
                noVideosEl.style.display = 'flex';
            } else {
                noVideosEl.style.display = 'none';
            }
        }

        function copyToClipboard(text, btn) {
            navigator.clipboard.writeText(text).then(() => {
                const originalContent = btn.innerHTML;
                btn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span style="color: #10b981;">Copied!</span>
                `;
                btn.style.background = 'rgba(16, 185, 129, 0.15)';
                btn.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                setTimeout(() => {
                    btn.innerHTML = originalContent;
                    btn.style.background = '';
                    btn.style.borderColor = '';
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    if not os.path.exists(VIDEO_DIR):
        return render_template_string(ERROR_TEMPLATE, dir=VIDEO_DIR)
    
    videos = []
    folders = set()
    
    for root, dirs, files in os.walk(VIDEO_DIR):
        rel_dir = os.path.relpath(root, VIDEO_DIR)
        if rel_dir == '.':
            rel_dir_display = 'Root'
        else:
            rel_dir_display = rel_dir
            
        dir_videos_exist = False
        for f in sorted(files):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                rel_path = os.path.relpath(os.path.join(root, f), VIDEO_DIR)
                videos.append({
                    'rel_path': rel_path,
                    'filename': f,
                    'folder': rel_dir_display
                })
                dir_videos_exist = True
                
        if dir_videos_exist:
            folders.add(rel_dir_display)
            
    # Sort videos by folder name, then filename
    videos.sort(key=lambda x: (x['folder'], x['filename']))
    
    # Calculate folder counts
    folder_counts = {}
    for v in videos:
        folder_counts[v['folder']] = folder_counts.get(v['folder'], 0) + 1
        
    # Sort folders list, ensuring Root is first
    sorted_folders = sorted(list(folders))
    if 'Root' in sorted_folders:
        sorted_folders.remove('Root')
        sorted_folders.insert(0, 'Root')
        
    dir_name = os.path.basename(VIDEO_DIR)
    if not dir_name:
        dir_name = VIDEO_DIR
        
    return render_template_string(
        HTML_TEMPLATE,
        videos=videos,
        folders=sorted_folders,
        folder_counts=folder_counts,
        dir_name=dir_name,
        dir_path=VIDEO_DIR
    )

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=args.port, host='0.0.0.0')
