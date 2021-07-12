# vehicle_reid
## Get Started
1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/slpluan/vehicle_reid.git`

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - cv2 (optional)


## Train

```bash
python train.py
```

## Test

```bash
python test.py
```
To get visualized reID results, first create `results` folder in log dir, then:
```bash
python ./get_vis_result.py

```
