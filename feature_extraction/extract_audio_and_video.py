import argparse
import os
import os.path as P
from functools import partial
from glob import glob
from multiprocessing import Pool


def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline(video_path, output_dir, fps, sr, duration_target):
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")

    # Extract Original Audio
    ori_audio_dir = P.join(output_dir, "audio_ori")
    os.makedirs(ori_audio_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -loglevel error -f wav -vn -y {P.join(ori_audio_dir, audio_name)}")

    # Cut Video According to Audio
    align_video_dir = P.join(output_dir, "videos_algin")
    os.makedirs(align_video_dir, exist_ok=True)
    duration = execCmd(f"ffmpeg -i {P.join(ori_audio_dir, audio_name)}  2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//")
    duration = duration.replace('\n', "")
    os.system("ffmpeg -ss 0 -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(
            duration, video_path, P.join(align_video_dir, video_name)))

    # Repeat Video
    repeat_video_dir = P.join(output_dir, "videos_repeat")
    os.makedirs(repeat_video_dir, exist_ok=True)
    hour, min, sec = [float(_) for _ in duration.split(":")]
    duration_second = 3600*hour + 60*min + sec
    n_repeat = duration_target//duration_second + 1
    os.system("ffmpeg -stream_loop {} -i {} -loglevel error -c copy -fflags +genpts -y {}".format(n_repeat,
            P.join(align_video_dir, video_name), P.join(repeat_video_dir, video_name)))

    # Cut Video
    cut_video_dir = P.join(output_dir, f"videos_{duration_target}s")
    os.makedirs(cut_video_dir, exist_ok=True)
    os.system("ffmpeg -ss 0 -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(duration_target,
            P.join(repeat_video_dir, video_name), P.join(cut_video_dir, video_name)))

    # Extract Audio
    cut_audio_dir = P.join(output_dir, f"audio_{duration_target}s")
    os.makedirs(cut_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -f wav -vn -y {}".format(
            P.join(cut_video_dir, video_name), P.join(cut_audio_dir, audio_name)))

    # change audio sample rate
    sr_audio_dir = P.join(output_dir, f"audio_{duration_target}s_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -ac 1 -ab 16k -ar {} -y {}".format(
            P.join(cut_audio_dir, audio_name), sr, P.join(sr_audio_dir, audio_name)))

    # change video fps
    fps_audio_dir = P.join(output_dir, f"videos_{duration_target}s_{fps}fps")
    os.makedirs(fps_audio_dir, exist_ok=True)
    os.system("ffmpeg -y -i {} -loglevel error -r {} -c:v libx264 -strict -2 {}".format(
              P.join(cut_video_dir, video_name), fps, P.join(fps_audio_dir, video_name)))


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/VAS/dog/videos")
    paser.add_argument("-o", "--output_dir", default="data/features/dog")
    paser.add_argument("-d", "--duration", type=int, default=10)
    paser.add_argument("-a", '--audio_sample_rate', default='22050')
    paser.add_argument("-v", '--video_fps', default='21.5')
    paser.add_argument("-n", '--num_worker', type=int, default=32)
    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    duration_target = args.duration
    sr = args.audio_sample_rate
    fps = args.video_fps

    video_paths = glob(P.join(input_dir, "*.mp4"))
    video_paths.sort()

    with Pool(args.num_worker) as p:
        p.map(partial(pipline, output_dir=output_dir,
        sr=sr, fps=fps, duration_target=duration_target), video_paths)
