# Object Detection Flask API

Custom object detection PyTorch yoloV3 pretrained model(Drone Detection)

## Installation

### Python Version

- Python == 3.7

### Library Installation

#### Windows

- Virtual Environment
  - `python -m venv venv`
  - `.\venv\Scripts\activate`
  - If any problem for scripts activation
    - Execute following command in administration mode
      - `Set-ExecutionPolicy Unrestricted -Force`
    - Later you can revert the change
      - `Set-ExecutionPolicy restricted -Force`
- Library Install
  - `python .\installation\get-pip.py`
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`

#### Linux

- Virtual Environment
  - `python -m venv venv`
  - `source venv/bin/activate`
- Library Install
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`

#### Pretrained Weight Download
- Download pretrained model from Follwing Link:
    - [Google Drive](https://drive.google.com/file/d/1j_4sQea3y3-MwrDLq5goUNqCGlGsl9hv/view?usp=sharing)
- Place in the **ml/weights** folder 

## Drone Detection


### Web Interface View

<table>
<tr align='center'>
<td><img src="README Files/a.png" alt="Male.jpg" width="800" height="400"/></td>
</tr align='center'>
<tr>
<td><img src="README Files/b.png" alt="Male.jpg" width="800" height="400"/></td>
</tr>
<table>

### Detection from Web Interface
- Run flask app
    - `python main.py`
- Open web interface
    - [http://0.0.0.0:8000](http://0.0.0.0:8000)
- Upload image from anywhere
    - Allowed image extensions [jpg, png, jpeg]

## Referecne

- [PyTorch yoloV3 Implementation](https://github.com/eriklindernoren/PyTorch-YOLOv3)