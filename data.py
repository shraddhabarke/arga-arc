import os
import json
import defs

ARC_PATH = os.path.join(defs.PROJECT_ROOT, "ARC/data")

def get_task_paths(filter=None, split=None):
    """
    Returns a list of paths to the json files corresponding to the tasks in ARC.
    `filter` is custom logic to select which tasks to include, if None all tasks are included.
    """
    paths = []
    if split is not None:
        assert split in ["training", "evaluation"]
        subsets = [split]
    else:
        subsets = ["training", "evaluation"]
    for subset in subsets:
        subset_path = os.path.join(ARC_PATH, subset)
        for task in os.listdir(subset_path):
            task_path = os.path.join(subset_path, task)
            if filter is None or filter(task_path):
                paths.append(task_path)
    return paths

def to_dict(task_paths):
    """
    Given a list of paths to json files representing ARC tasks, returns a dictionary
    mapping task ids to the corresponding json objects.
    """
    task_dict = {}
    for task_path in task_paths:
        task = json.load(open(task_path))
        task_id = os.path.basename(task_path).split(".")[0]
        task_dict[task_id] = task
    return task_dict

def all_tasks_dict(split=None):
    """
    Return a dictionary indexed by task_id with all (800) the tasks in ARC,
    or in a given split
    """
    return to_dict(get_task_paths(split=split))

def get_tasks_dict(task_ids):
    """
    Given a list of task ids, returns a dictionary mapping task ids to the corresponding json objects.
    """
    all_tasks = all_tasks_dict()
    return { task_id : all_tasks[task_id] for task_id in task_ids }

def get_task(task_id):
    return all_tasks_dict()[task_id]

# for ARCAM
def arcam_all_tasks_dict():
    """
    Return a dictionary indexed by task_id with all the tasks in ARCAM
    """
    arcam_path = os.path.join(defs.PROJECT_ROOT, "ARCAM/tasks")
    task_dict = {}
    for task in os.listdir(arcam_path):
        task_path = os.path.join(arcam_path, task)
        task = json.load(open(task_path))
        task_id = os.path.basename(task_path).split(".")[0]
        task_dict[task_id] = task
    return task_dict

def arcam_get_task(task_id):
    return arcam_all_tasks_dict()[task_id]

def pretty_grid(grid, color_map="id"):
    color_ids = range(10)
    if color_map == "id":
        color_map = { i : str(i) for i in color_ids }
    elif color_map == "nobg":
        map = lambda x: " " if x == 0 else str(x)
        color_map = { i : map(i) for i in color_ids }
    elif color_map == "char":
        map = ["O", "B", "R", "G", "Y", "X", "F", "A", "C", "W"]
        color_map = { i : map[i] for i in color_ids }
    pretty_row = lambda row: " ".join(color_map[x] for x in row)
    return "\n".join(pretty_row(row) for row in grid)

def task_description(task_id, print_test=False, color_map="id"):
    task = get_task(task_id)
    desc = []
    for i, ex in enumerate(task["train"]):
        desc.append(f"EXAMPLE {i + 1}")
        desc.append("INPUT:")
        desc.append(pretty_grid(ex["input"], color_map=color_map))
        desc.append("OUTPUT:")
        desc.append(pretty_grid(ex["output"], color_map=color_map))
        desc.append("")
    if print_test:
        for i, tst in enumerate(task["test"]):
            desc.append(f"TEST {i + 1}")
            desc.append("INPUT:")
            desc.append(pretty_grid(tst["input"], color_map=color_map))
            desc.append("")
    return "\n".join(desc)