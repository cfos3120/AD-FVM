import torch
from torch.utils.data import DataLoader 
from typing import Tuple
import numpy as np
import time
import os
import subprocess
import sys
from .models.gnot_custom import Custom_GNOT

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        num_cells_i = np.argmax(x.shape)

        #Assume uniform 2d mesh
        h = 1.0 / (int(np.sqrt(x.shape[num_cells_i])) - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y=None):
        # Custom adjustment here, if compared to zero we do absolute L2
        if y is None:
            return self.abs(x, torch.zeros_like(x))
        elif torch.all(y == 0):
            return self.abs(x, y)
        else:
            return self.rel(x, y)
        

def get_GNOT_model(dataset_params, params):
    if params['model_name'] == 'GNOT':
        from .models.gnot_custom import Custom_GNOT as GNOT
    else:
        from .models.geneva import GenevaNOT as GNOT

    model = GNOT(trunk_size=dataset_params['input_dim'],
                    branch_sizes=dataset_params['branch_sizes'],
                    space_dim=dataset_params['space_dim'],
                    output_size=dataset_params['output_dim'],
                    n_head=params['n_head'],
                    n_layers=params['n_layers'],
                    n_experts=params['n_experts'],
                    n_hidden=params['n_hidden'],
                    attn_dropout=params['attn_dropout'],
                    attn_type=params['attn_type'],
                    **{k: params[k] for k in ['rnn_input_i', 'rollout_steps', 'in_timesteps'] if k in params}
                    )
    
    return model


def basic_re_n_dataset_split(dataset_params)-> Tuple[Tuple[torch.Tensor]]:

    data_path = dataset_params['dataset_dir']
    data_dict   = np.load(data_path,allow_pickle=True).item()
    assert 'Re' in list(data_dict.keys())
    Re = torch.tensor(data_dict['Re'])
    x = torch.tensor(data_dict['Points'])
    y = torch.tensor(data_dict['Solutions'])
    n_internal = data_dict['Points'].shape[0]
    n_cases = data_dict['Solutions'].shape[0]

    print(f'Reynolds Number Range: {Re.min()}-{Re.max()}')
    print(f'Total Cases: {n_cases}')
    print(f'Total Internal Cells: {n_internal}')

    # Split for training - testing
    train_ratio = dataset_params['data_split']
    n_batches = y.shape[0]
    train_size = int(train_ratio * n_batches)
    test_size = n_batches - train_size

    seed_generator = torch.Generator().manual_seed(42)

    # Pin the start and end of cohort as training data for a purely interpolative model
    train_split, test_split = torch.utils.data.random_split(y[1:-1,...], [train_size-2, test_size], generator=seed_generator)
    test_split.indices = list(1 + np.array(test_split.indices))
    train_split.indices = list(1 + np.array(train_split.indices))
    train_split.indices.append(0)
    train_split.indices.append(-1)

    train_dataset,  test_dataset = y[train_split.indices,...], y[test_split.indices,...]
    train_Re,    test_Re         = Re[train_split.indices], Re[test_split.indices]
    print(f'Dataset Split up for training using torch generator seed: {seed_generator.initial_seed()}')
    
    return [x, train_dataset, train_Re], [x, test_dataset, test_Re]

class Basic_gnot_batch_sampler(torch.utils.data.Dataset):
    def __init__(self,dataset:Tuple):
        super(Basic_gnot_batch_sampler, self).__init__()
        self._x:torch.Tensor = dataset[0]
        self._y:torch.Tensor = dataset[1]
        self._Re:torch.Tensor = dataset[2]

    def __len__(self):
        return len(self._Re)
    
    def collate_function(self):
        return None

    def __getitem__(self, idx):
        in_keys     = self._Re[idx].float().reshape(1,1)
        in_queries  = self._x.float()
        out_truth   = self._y[idx].float()
        input_f     = [in_keys.float()]
        
        return in_queries, input_f, out_truth

def get_dataset_loaders(params, dataset_params)->Tuple[DataLoader]:
    train_dataset, val_dataset = basic_re_n_dataset_split(dataset_params)
    train_sampler = Basic_gnot_batch_sampler(train_dataset)
    val_sampler = Basic_gnot_batch_sampler(val_dataset)

    train_loader = DataLoader(dataset=train_sampler,
                            batch_size=4,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(params['seed']))
    val_loader = DataLoader(dataset=val_sampler,
                            batch_size=4,
                            shuffle=False,
                            generator=torch.Generator().manual_seed(params['seed']))
    return train_loader, val_loader

def create_save_dir(config:dict) -> str:
    
    # set timestamp
    config['run_info'] = {}
    config['run_info']['timestamp'] = time.strftime('%Y%m%d_%Hh%Mm')

    model_type = config['parameters'].get('model_name', 'Unknown')
    problem_type = config['dataset_settings'].get('name', 'Unknown')

    out_dir = f'./tests/models/trained_models/{model_type}_{problem_type}_{config["run_info"]["timestamp"]}'
    config['run_info']['out_dir'] = out_dir

    # Try fetching the git_id
    try:
        tag = subprocess.check_output(
                ["git", "describe", "--always"],
                text=True,
                stderr=subprocess.STDOUT,
                ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        tag = 'Unknown'
        print(f"Could not determine git tag: {e}", file=sys.stderr)
    config['run_info']['git_commit_hash'] = tag

    # Make directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    return out_dir