# Image Fusion

## Requirements

The program requires python 3. Besides, use `pip3 instsall -r requirements.txt` to install dependent packages.

## Usage

`python3 image_fusion.py -h` to get params info.

Most params are clear, and some further explainations are as follow:

1. `-t --times`: the program uses Gauss-Seidel method to solve Ax = b. This param sets iteration times.
2. `-h --height` and `-w --width`: position of mask picture's (0, 0) in background. Accept negative integer(s). You MUST make sure that masked area is totally legal in background with given position.

## Demo

```shell
python3 image_fusion.py -i test1_src.jpg -o test1_result_1000.jpg -m test1_mask.jpg -b test1_target.jpg -t 1000 -y -159 -x -93
```
