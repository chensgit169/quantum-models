## Quantum Models

This repository contains some codes and notes from my study and research on quantum physics. I am glad to share and discuss relevant ideals and methods :)

```python
# Under building...
```

Chen Wei, weichen191@mails.ucas.ac.cn



## Installation

The easiest way to download the project is via `git`. Firstly, navigate to the directory where you aim to place the repository, say `projects`, and run
```bash
git clone https://github.com/chensgit169/quantum-models
```
Once the cloning process is complete, a new folder `projects/quantum-models`will be created.

It is recommended to create a Python virtual environment. Navigate to the directory  `quantum_models` and run
```bash
python -m venv venv
```
(On Linux you may need to replace `python` command with `python3`.) 

To activate the virtual environment, on Linux/macOS:
```bash
source venv/bin/activate
```
On Windows using Git Bash:
```bash
source venv/Scripts/activate
```
After activation, `(venv)` should appear at the beginning of the command prompt. To leave the virtual environment, simply run
```
deactivate
```
The virtual environment isolates all the Python dependencies for the present project and helps avoid conflicts with other files in your system. 

The relied on packages have been specified in `pyproject.toml`. Finally, install the project by executing
```bash
pip install -e .
```
inside the `quantum-models` directory. Here, `-e` stands for *editable*, meaning that changes to the source code take effect immediately without requiring reinstallation. This command also ensures that the project's Python modules (for example `su2`) are discoverable from within the environment.
