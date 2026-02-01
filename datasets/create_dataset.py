from datasets import load_dataset
#d#ataset = load_dataset("json", data_files="train/derp.json")
dataset = load_dataset("json", data_files={"train": "train/derp.json", "test": "test/derp.json"})
dataset.save_to_disk("saved_local_dataset")
