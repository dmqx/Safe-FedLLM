import random

def split_dataset(_fed_args, script_args, dataset, num_client):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    splited_datasets = [dataset.shard(num_client, i) for i in range(num_client)]
    return splited_datasets

def get_dataset_this_round(dataset, round, _fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round
