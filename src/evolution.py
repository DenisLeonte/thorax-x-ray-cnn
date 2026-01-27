import random
import copy

def get_default_config():
    return {
        "blocks": [
            {"filters": 32, "kernel": 3, "pool": True},
            {"filters": 64, "kernel": 3, "pool": True},
            {"filters": 128, "kernel": 3, "pool": True},
            {"filters": 256, "kernel": 3, "pool": True},
            {"filters": 512, "kernel": 3, "pool": True}
        ],
        "fc_layers": [512],
        "dropout": 0.5,
        "lr": 0.0001,
        "batch_size": 64  # Start with higher batch size for GPU utilization
    }

def mutate_config(config):
    """
    Randomly mutates the configuration to explore new architectures.
    Returns a DEEP COPY of the mutated config.
    """
    new_config = copy.deepcopy(config)
    
    mutation_type = random.choice([
        "add_block", "remove_block", "change_filters", 
        "change_dropout", "change_lr", "change_fc",
        "change_batch_size"
    ])
    
    print(f"--> Evolution Strategy: {mutation_type}")

    if mutation_type == "add_block":
        # Add a block similar to the last one, but maybe more filters
        last_block = new_config['blocks'][-1]
        new_filters = min(512, last_block['filters'] * 2)
        new_config['blocks'].append({"filters": new_filters, "kernel": 3, "pool": True})
        
    elif mutation_type == "remove_block":
        if len(new_config['blocks']) > 2:
            new_config['blocks'].pop()
            
    elif mutation_type == "change_filters":
        # Pick a random block and change its filters
        idx = random.randint(0, len(new_config['blocks']) - 1)
        choices = [32, 64, 128, 256, 512]
        new_config['blocks'][idx]['filters'] = random.choice(choices)
        
    elif mutation_type == "change_dropout":
        new_config['dropout'] = random.choice([0.2, 0.3, 0.4, 0.5, 0.6])
        
    elif mutation_type == "change_lr":
        # Perturb LR slightly
        new_config['lr'] = new_config['lr'] * random.choice([0.5, 0.8, 1.2, 2.0])
        new_config['lr'] = max(1e-5, min(1e-3, new_config['lr']))
        
    elif mutation_type == "change_fc":
        # Change size of first FC layer
        new_config['fc_layers'][0] = random.choice([256, 512, 1024, 2048])

    elif mutation_type == "change_batch_size":
        new_config['batch_size'] = random.choice([32, 64, 128])
        
    return new_config
