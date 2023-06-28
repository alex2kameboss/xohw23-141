# AI-augumented barcode reader using Xilinx Kria KV260
Repository for Xilinx OpenHW Competition 2023. Team from Transylvania University from Brasov.

## Requirement:
- Vitis 2021.2
- Vivado 2021.2
- [Vitis AI 1.4](https://github.com/Xilinx/Vitis-AI/tree/1.4)
- Kria KV260 Vision AI + [Ubuntu 22.04](https://ubuntu.com/download/amd-xilinx)
- [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ)
- Python 3.10
- OpenCV 4.6
- USB webcam

## Folder structure:
- **kv260_base_platform** - has the script to automaticaly build KV260 base platform and the *xsa* file for vitis
- **vitis** - has config file for the DPU, also her are available the output files: bistream, hardware handoff and *xclib*
- **software_prototype** - the scripts for first prototype, this run on CPU
- **board** - script and files for the KV260

## Run demo
1. run KV260 with Ubuntu 22.04, [doc](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/2363129857/Getting+Started+with+Certified+Ubuntu+22.04+LTS+for+Xilinx+Devices)
2. install Kria-PYNQ, [doc](https://github.com/Xilinx/Kria-PYNQ#2-install-pynq)
3. clone this repo, *git clone --recurse-submodules -j8 https://github.com/alex2kameboss/xohw23-141.git*
4. connect with ssh X11 forwarding at KV260
### On ssh session
5. add X11 forwarding to root
```bash
xauth list
# output in format alex-Legion/unix:0  MIT-MAGIC-COOKIE-1  37e6a5fbc24235c71aeb63c4b16e346b, copy it
sudo su
export DISPLAY=localhost:10
xauth add <result form xauth list>
```
6. cd xohw23-141/board
7. connect USB webcam
8. python3 main_sequencial.py
