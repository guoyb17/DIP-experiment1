# Face Morphing

## Requirements

The program requires python 3. Besides, use `pip3 instsall -r requirements.txt` to install dependent packages.

The program uses Face++ API to mark faces. Your own API token and secret is required. Create `API_SECRET` file in this directory, which should look like:

```
Token123ABCabcSecret
Secret123ABCabcToken
```

When running, network is necessary. Your input photos will be sent to Face++, which has completely **NO RELATIONSHIP** with me. **DO NOT RUN** or use your own photos if you concern about privacy.

Also, animal faces is not supported due to API function. You need to manually mark points and modify program to run without API restriction.

## Usage

`python3 image_fusion.py -h` to get params info.

## Demo

```shell
python3 face_morphing.py -f source1.png -t target1.png -p result1 -n 3
```
