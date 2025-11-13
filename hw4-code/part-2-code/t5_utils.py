import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''
    Initialize Weights & Biases for experiment tracking.

    Args:
        args: Arguments containing wandb configuration (project name, run name, etc.)
    '''
    if hasattr(args, 'use_wandb') and args.use_wandb:
        # Initialize wandb
        wandb.init(
            project=args.wandb_project if hasattr(args, 'wandb_project') else 'nlp-t5-training',
            name=args.wandb_run_name if hasattr(args, 'wandb_run_name') else None,
            config=vars(args),  # Log all hyperparameters
            reinit=True
        )
        print(f"Weights & Biases initialized for project: {wandb.run.project}")
    else:
        print("Weights & Biases tracking is disabled")

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        # Load pretrained model with weights
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        # Load config and initialize model from scratch (random weights)
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)

    # Move model to device
    model = model.to(DEVICE)

    # Optionally freeze parameters
    if hasattr(args, 'freeze_encoder') and args.freeze_encoder:
        # Freeze encoder parameters
        for param in model.encoder.parameters():
            param.requires_grad = False

    if hasattr(args, 'freeze_decoder') and args.freeze_decoder:
        # Freeze decoder parameters
        for param in model.decoder.parameters():
            param.requires_grad = False

    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint to be able to load the model later.

    Args:
        checkpoint_dir: Directory to save the checkpoint
        model: The T5 model to save
        best: Boolean indicating if this is the best model (True) or last model (False)
    '''
    # Create checkpoint directory if it doesn't exist
    mkdir(checkpoint_dir)

    # Determine filename based on whether this is the best or last checkpoint
    filename = 'best_model.pt' if best else 'last_model.pt'
    filepath = os.path.join(checkpoint_dir, filename)

    # Save model state dict
    torch.save(model.state_dict(), filepath)

    print(f"Model saved to {filepath}")

def load_model_from_checkpoint(args, best):
    '''
    Load model from a checkpoint.

    Args:
        args: Arguments containing checkpoint_dir and other model configuration
        best: Boolean indicating whether to load the best model (True) or last model (False)

    Returns:
        model: The loaded T5 model
    '''
    # Determine filename based on whether to load best or last checkpoint
    filename = 'best_model.pt' if best else 'last_model.pt'
    filepath = os.path.join(args.checkpoint_dir, filename)

    # Check if checkpoint exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    # Initialize model architecture (same as in initialize_model)
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)

    # Load state dict
    state_dict = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Move model to device
    model = model.to(DEVICE)

    print(f"Model loaded from {filepath}")

    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

