# Demo

## Install
Please install additional libraries by running:
```bash
pip install -r requirements.txt
```

To work on video, you need to install `ffmpeg`
```
sudo apt-get install ffmpeg
```

## Set up XMem (Deprecated)
```bash
git clone https://github.com/hkchengrex/XMem.git
cd XMem
pip install -r requirements.txt
```

## Set up Samurai (Extension of SAM2.1)

Clone and install [Samurai](https://github.com/educelab/samurai)

Overwrite the file to make it run with multiple instances:
```
cp sam2_base.py /samurai/sam2/sam2/modeling/sam2_base.py
```

## Running
To run demo:
```bash
gradio app.py
```

to run old XMem:
```bash
RUN_XMEM=1 gradio app.py
```