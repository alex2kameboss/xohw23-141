# xohw23-141
Repository for Xilinx OpenHW Competition 2023. Team from Transylvania University from Brasov.

Requirement:
- Vitis 2021.2
- Vivado 2021.2
- [Vitis AI 1.4](https://github.com/Xilinx/Vitis-AI/tree/1.4)
- Kria KV260 Vision AI + [Ubuntu 22.04](https://ubuntu.com/download/amd-xilinx)
- [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ)
- Python 3.10
- OpenCV 4.6
- vitis-ai-runtime 2.5

Folder structure:
- **kv260_base_platform** - has the script to automaticaly build KV260 base platform and the *xsa* file for vitis
- **vitis** - has config file for the DPU, also her are available the output files: bistream, hardware handoff and *xclib*
- **software_prototype** - the scripts for first prototype, this run on CPU
- **board** - script and files for the KV260
