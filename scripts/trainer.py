# Import Neptune
import neptune.new as neptune
import os

# Initialize the Neptune objects you need
run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),  # get from your repo secret
    project="workspace-name/project-name",  # replace with your own
)

# Define and log metadata
PARAMS = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam",
}

run["parameters"] = PARAMS

...
