import argparse
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Concatenate videos horizontally from multiple folders.")
    parser.add_argument("--in_folder", nargs='+', required=True, help="List of input folders")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    if not args.in_folder:
        raise ValueError("Provide at least one input folder.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Get method names from deepest subfolder
    method_names = [os.path.basename(os.path.normpath(f)) for f in args.in_folder]
    prefix = "#".join(method_names)

    # Find common videos
    video_exts = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}
    common_videos = None

    for folder in args.in_folder:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        
        files = {f for f in os.listdir(folder) if Path(f).suffix.lower() in video_exts}
        if common_videos is None:
            common_videos = files
        else:
            common_videos = common_videos.intersection(files)

    if not common_videos:
        print("No common videos found across all input folders.")
        return

    print(f"Found {len(common_videos)} common video(s).")

    for vid in common_videos:
        inputs = []
        filter_inputs = ""
        for i, folder in enumerate(args.in_folder):
            vid_path = os.path.join(folder, vid)
            inputs.extend(["-i", vid_path])
            # Scale each video to make sure they have the exact same height before stacking
            # (hstack requires inputs to have identical heights)
            filter_inputs += f"[{i}:v]"

        out_name = f"{prefix}_{vid}"
        out_path = os.path.join(args.out_dir, out_name)

        cmd = ["ffmpeg", "-y"] + inputs
        
        if len(args.in_folder) > 1:
            filter_complex = f"{filter_inputs}hstack=inputs={len(args.in_folder)}[v]"
            # Map stacked video, copy audio from the first video if it exists
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-c:a", "aac"
            ])
        else:
            cmd.extend(["-c", "copy"])
            
        cmd.append(out_path)

        print(f"Processing: {out_name}")
        try:
            # Hide ffmpeg overwhelming output but keep errors
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {vid}. FFmpeg stderr:\n{e.stderr}")

if __name__ == "__main__":
    main()
