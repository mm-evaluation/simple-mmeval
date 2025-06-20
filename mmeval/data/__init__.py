from .local_loader import LocalDatasetLoader
from .huggingface_loader import MMEDatasetLoader, MMVetDatasetLoader, SeedBenchDatasetLoader

def load_dataset(args):
    if args.dataset == "local@json":
        loader = LocalDatasetLoader(
            data_file=getattr(args, 'infile', None),
            img_dir=getattr(args, 'img_dir', None)
        )
        loader.load(
            sample_mode=getattr(args, 'sample_mode', 'all'),
            sample_number=getattr(args, 'sample_number', None)
        )
        return loader.dataset
    
    elif args.dataset == "MME":
        loader = MMEDatasetLoader(
            dataset_dir=getattr(args, 'dataset_dir', None),
            hf_home=getattr(args, 'hf_home', None)
        )
        loader.load(
            split=getattr(args, 'split', 'all'),
            sample_mode=getattr(args, 'sample_mode', 'all'),
            sample_number=getattr(args, 'sample_number', None)
        )
        return loader.dataset
    
    elif args.dataset == "MM-Vet":
        loader = MMVetDatasetLoader(
            dataset_dir=getattr(args, 'dataset_dir', None),
            hf_home=getattr(args, 'hf_home', None)
        )
        loader.load(
            split=getattr(args, 'split', 'all'),
            sample_mode=getattr(args, 'sample_mode', 'all'),
            sample_number=getattr(args, 'sample_number', None)
        )
        return loader.dataset
    
    elif args.dataset == "SEED-Bench":
        loader = SeedBenchDatasetLoader(
            dataset_dir=getattr(args, 'dataset_dir', None),
            hf_home=getattr(args, 'hf_home', None)
        )
        loader.load(
            split=getattr(args, 'split', 'all'),
            sample_mode=getattr(args, 'sample_mode', 'all'),
            sample_number=getattr(args, 'sample_number', None)
        )
        return loader.dataset
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")