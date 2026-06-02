import argparse
import os
import subprocess
import glob
from pathlib import Path

def get_unique_method_names(folders):
    """
    Extracts the shortest unique identifying names for a list of folder paths
    by stripping common prefix and suffix components.
    """
    if len(folders) <= 1:
        return [os.path.basename(os.path.normpath(f)) for f in folders]
        
    abs_paths = [os.path.abspath(os.path.normpath(f)) for f in folders]
    path_parts = [p.split(os.sep) for p in abs_paths]
    
    # Remove common prefix parts
    min_len = min(len(parts) for parts in path_parts)
    common_prefix_len = 0
    for i in range(min_len):
        if len(set(parts[i] for parts in path_parts)) == 1:
            common_prefix_len += 1
        else:
            break
            
    # Remove common suffix parts
    common_suffix_len = 0
    for i in range(1, min_len - common_prefix_len + 1):
        if len(set(parts[-i] for parts in path_parts)) == 1:
            common_suffix_len += 1
        else:
            break
            
    method_names = []
    for parts in path_parts:
        start_idx = common_prefix_len
        end_idx = len(parts) - common_suffix_len
        if start_idx >= end_idx:
            start_idx = max(0, len(parts) - 1)
            end_idx = len(parts)
        unique_parts = parts[start_idx:end_idx]
        method_names.append("_".join(unique_parts))
        
    return method_names

def main():
    parser = argparse.ArgumentParser(description="Concatenate videos horizontally from multiple folders.")
    parser.add_argument("--in_folder", nargs='+', required=True, help="List of input folders or glob patterns")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    # Expand glob patterns
    expanded_folders = []
    for pattern in args.in_folder:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded_folders.extend(matches)
        else:
            expanded_folders.append(pattern)
    args.in_folder = expanded_folders

    if not args.in_folder:
        raise ValueError("Provide at least one input folder.")

    # Check existence of folders (skip invalid ones with a warning)
    valid_folders = []
    for folder in args.in_folder:
        if os.path.isdir(folder):
            valid_folders.append(folder)
        else:
            print(f"Warning: Directory not found, skipping: {folder}")
    
    args.in_folder = valid_folders
    if not args.in_folder:
        raise ValueError("No valid input folders found to process.")

    # Detect if we should process subfolders automatically
    video_exts = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}
    folder_files = {}
    common_videos_direct = None
    for folder in args.in_folder:
        files = {f for f in os.listdir(folder) if Path(f).suffix.lower() in video_exts}
        folder_files[folder] = files
        if common_videos_direct is None:
            common_videos_direct = files
        else:
            common_videos_direct = common_videos_direct.intersection(files)

    jobs = []
    if common_videos_direct:
        jobs.append(("", args.in_folder, common_videos_direct))
    else:
        # No common videos directly, look for common subdirectories
        folder_subdirs = {}
        common_subdirs = None
        for folder in args.in_folder:
            subdirs = {d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))}
            folder_subdirs[folder] = subdirs
            if common_subdirs is None:
                common_subdirs = subdirs
            else:
                common_subdirs = common_subdirs.intersection(subdirs)
        
        if common_subdirs:
            print(f"No common videos found in input roots. Processing common subfolders: {', '.join(sorted(common_subdirs))}")
            for sub in sorted(common_subdirs):
                sub_folders = [os.path.join(f, sub) for f in args.in_folder]
                sub_folder_files = {}
                common_vids = None
                for s_folder in sub_folders:
                    files = {f for f in os.listdir(s_folder) if Path(f).suffix.lower() in video_exts}
                    sub_folder_files[s_folder] = files
                    if common_vids is None:
                        common_vids = files
                    else:
                        common_vids = common_vids.intersection(files)
                if common_vids:
                    jobs.append((sub, sub_folders, common_vids))
                else:
                    print(f"\nNo common videos found in subfolder '{sub}'. Subfolder contents summary:")
                    for s_folder, files in sorted(sub_folder_files.items()):
                        print(f"  - {os.path.basename(os.path.dirname(s_folder))}/{sub}: {len(files)} video(s)")
        else:
            print("\nNo common subdirectories found across the input folders. Subdirectory summary:")
            for folder, subdirs in sorted(folder_subdirs.items()):
                print(f"  - {os.path.basename(folder)}: subdirectories={sorted(list(subdirs))}")
            # Fallback to direct folders to print a warning/error
            jobs.append(("", args.in_folder, set()))

    processed_count = 0

    for sub, folders, common_videos in jobs:
        if not common_videos:
            continue

        print(f"\nProcessing target '{sub if sub else 'root'}' with {len(common_videos)} common video(s)...")
        method_names = get_unique_method_names(folders)
        prefix = "#".join(method_names)

        out_dir = os.path.join(args.out_dir, sub) if sub else args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        for vid in sorted(list(common_videos)):
            inputs = []
            filter_inputs = ""
            for i, folder in enumerate(folders):
                vid_path = os.path.join(folder, vid)
                inputs.extend(["-i", vid_path])
                filter_inputs += f"[{i}:v]"

            out_name = f"{prefix}_{vid}"
            out_path = os.path.join(out_dir, out_name)

            cmd = ["ffmpeg", "-y"] + inputs
            
            if len(folders) > 1:
                filter_complex = f"{filter_inputs}hstack=inputs={len(folders)}[v]"
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

            print(f"  Concatenating into: {os.path.join(sub, out_name) if sub else out_name}")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                processed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  Error processing {vid}. FFmpeg stderr:\n{e.stderr}")

    print(f"\nDone! Concatenated {processed_count} videos.")

if __name__ == "__main__":
    main()
