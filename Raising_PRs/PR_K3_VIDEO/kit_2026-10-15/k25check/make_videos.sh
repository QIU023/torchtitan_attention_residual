#!/bin/bash
# The five test videos of compare_k25.py (10-18), from two cc12m-test photos and ffmpeg's test sources.
set -e
ffmpeg -hide_banner -loglevel error -y -loop 1 -i 000000006.jpg -vf "scale=3360:2000,crop=2560:1440:x='t*200':y='t*100'" -t 3 -r 24 -c:v libx264 -pix_fmt yuv420p pan_2560x1440_24.mp4
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=3840x2160:rate=30000/1001" -t 2 -c:v libx264 -pix_fmt yuv420p testsrc2_3840x2160_2997.mp4
ffmpeg -hide_banner -loglevel error -y -loop 1 -i 000000014.jpg -vf "scale=2000:3200,crop=1440:2560:x='t*150':y='t*200'" -t 2 -r 60 -c:v libx264 -pix_fmt yuv420p pan_1440x2560_60.mp4
ffmpeg -hide_banner -loglevel error -y -loop 1 -i 000000006.jpg -vf "crop=840:500" -t 3 -r 15 -c:v libx264 -pix_fmt yuv420p photo_840x500_15.mp4
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "mandelbrot=size=1280x720:rate=25" -t 2 -c:v libx264 -pix_fmt yuv420p mandelbrot_1280x720_25.mp4
