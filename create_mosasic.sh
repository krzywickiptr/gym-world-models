ENVNAME=$1
ffmpeg -i "$ENVNAME"_1.avi -i "$ENVNAME"_2.avi -i "$ENVNAME"_3.avi -filter_complex vstack=inputs=3 "$ENVNAME".avi