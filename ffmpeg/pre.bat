if exist convert\%1 (echo "exist convert/"%1) else (mkdir convert\%1)
ffmpeg -i %1.mp4 -vf fps=1/20  convert\%1\%1%%03d.png