# Species-Aware Fish Length Estimation from Images

TODO: documentation

## Setup

First, please install uv if you do not have this. Used for dependency and version management
```bash
# For mac/linux
curl -LsSf https://astral.sh/uv/install.sh | sh 

# For windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 
```
While in the project directory, run the following to sync your environment with needed dependencies
```bash
uv sync
```

Now to download the dataset, please use git lfs to manage and track these files
```bash
sudo apt update && sudo apt install git-lfs
git lfs install
```
Then pull the data
```bash
git lfs pull
```
## Testing

To perform a smoke test, please run
```bash
uv run pytest
```

