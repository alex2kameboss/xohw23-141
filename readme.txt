Team number: xohw23-141
Project name: AI-augumented barcode reader using Xilinx Kria KV260
Link to YouTube Video(s): https://youtu.be/-n1rPmGHM_w
Link to project repository: https://github.com/alex2kameboss/xohw23-141

 
University name: "Transilvania" University of Brasov 
Participant: Alexandru Puşcaşu
Email: alexandru.puscasu@student.unitbv.ro
Participant: Ioana Ailenei
Email: ioana.ailenei@student.unitbv.ro
Participant: Mihai Miu
Email: mihai.miu@student.unitbv.ro

Supervisor name: Cătălin Ciobanu
Supervisor e-mail: catalin.ciobanu@unitbv.ro

 
Board used: Kria KV260 Vision AI
Software Version: Vitis & Vivado 2021.2, Vitis AI 1.4.1
Brief description of project: In the last few years, self-check-out and automatic check-out solutions have gained interest in the retail industry. We design a solution that approaches the problem in a new and different manner. Our project consists of developing an AI-based application that uses a camera to read EAN-13 barcodes, track them, and automatically add them to the bill. This project aims to enhance FPGA capabilities for AI-accelerated low-power portable systems using the Xilinx Kria KV260. Using the power of AI, this project aims to streamline processes and improve efficiency in everyday life.


Description of archive (explain directory structure, documents and source files):
* kv260_base_platform - has the script to automaticaly build KV260 base platform and the xsa file for vitis
* vitis - has config file for the DPU, also her are available the output files: bistream, hardware handoff and xclib
* software_prototype - the scripts for first prototype, this run on CPU
* board - script and files for the KV260

Instructions to build and test project
Step 1: run KV260 with Ubuntu 22.04, https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/2363129857/Getting+Started+with+Certified+Ubuntu+22.04+LTS+for+Xilinx+Devices
Step 2: install Kria-PYNQ, https://github.com/Xilinx/Kria-PYNQ#2-install-pynq
Step 3: clone repo, git clone --recurse-submodules -j8 https://github.com/alex2kameboss/xohw23-141.git
Step 4: connect with ssh X11 forwarding at KV260
On ssh session
Step 5: add X11 forwarding to root
Code:
xauth list
# output in format alex-Legion/unix:0  MIT-MAGIC-COOKIE-1  37e6a5fbc24235c71aeb63c4b16e346b, copy it
sudo su
export DISPLAY=localhost:10
xauth add <result form xauth list>
Step 6: cd xohw23-141/board
Step 7: connect USB webcam
Step 8: python3 main_sequencial.py