# simple-mmeval

- Simple: adding models and dataset requires only a py file with a few lines of code
- Self contained: All inference code are self contained (we provide inference environment in docker images)
- Modular: evaluation process are broken into several steps, connected with a python written connector, without interference
- Easy to use: inference just in a few lines of command

# test

1. **Activate the Conda Environment**
   conda activate pytorch

2. **Navigate to the Project Directory**
   cd simple-mmeval-wenyuan

3. **Run the Script**
   bash scripts/test_bed/qwen2d5.sh

