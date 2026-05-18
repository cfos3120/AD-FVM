import torch
import numpy as np
import argparse
from pathlib import Path
import yaml

from src import OpenFoam_Mesh
from src import Physics_loss_controller
from .utils import LpLoss, get_GNOT_model, get_dataset_loaders

def legacy_load_model(checkpoint_file_path, config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    ckpt = torch.load(checkpoint_file_path)
    return  ckpt['model_state_dict'], config

def load_model(checkpoint_file_path):
    ckpt = torch.load(checkpoint_file_path)
    print('Loading Model with Training Run Info:')
    for k,v in ckpt['config']['run_info'].items(): print(' ', k, v)
    return ckpt['model_state_dict'], ckpt['config']

# load configuration from json
parser = argparse.ArgumentParser("TRAIN THE FLASH GNOT TRANSFORMER")
parser.add_argument('--ckpt_path', type=str, help=".pth model checkpoint file", default='./tests/models/configs/cylinder_2d.yaml')
parser.add_argument('--config_path', type=str, help="model config file (legacy method)", default=None)
args = parser.parse_args()

def validate(ckpt, config):
    params = config['parameters']
    dataset_params = config['dataset_settings']

    # Override for consistency
    


    # Prepare Dataset
    train_dataloader, test_dataloader = get_dataset_loaders(params, dataset_params)

    # Setup FVM and Physics-Loss Controller
    Mesh = OpenFoam_Mesh(openfoam_case_dir=Path(dataset_params['openFOAM_case_dir']), 
                         dtype=torch.float32, 
                         corrected=True)
    Mesh._print_mesh_components()

    # Setup Navier-Stokes loss calculator
    assert Mesh.dim == 2
    Physics = Physics_loss_controller(physics_fn='ns_dns_incompress', 
                                      loss_fn=LpLoss(reduction=False), 
                                      channel_dict={'U':[0,1], 'p':[2]}, # 2D mesh
                                      mesh=Mesh)

    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_GNOT_model(dataset_params=dataset_params, params=params)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Move FVM framework to device
    Mesh.to(device)

    # Load model
    print('Loading in Model Weights')
    model.load_state_dict(ckpt)
    model.eval()

    # Key Metric:
    Loss_fn = LpLoss(reduction=False)
    
    # Total output storage (for larger datasets, this is best not stored in memory)
    model_outputs = {'Pred':[], 'Re':[], 'Pred_L2':[], 'PDE':{}, 'PDE_L2':{}, 'Test_f':[]}
    
    # Infer solutions (Subroutine)
    def eval_batch(i, batch, test=False): 
        with torch.no_grad():
            x, input_f, y = batch
            x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)
            out = model(x=x, inputs=input_f)

            # Sample info
            model_outputs['Re'].append(input_f[0].cpu().numpy())
            if test:
                model_outputs['Test_f'].append(np.ones(x.shape[0]))
            else:
                model_outputs['Test_f'].append(np.zeros(x.shape[0]))

            # Prediction metrics
            model_outputs['Pred'].append(out.cpu().numpy())
            model_outputs['Pred_L2'].append(Loss_fn(out,y).cpu().numpy())

            # PDE metrics
            pde_losses = Physics.compute(out.unsqueeze(1), Re=input_f[0], volume_weighted=True)
            
            if i == 0:
                model_outputs['PDE'] = {k:[v] for k,v in Physics.loss_dict.items()}
                model_outputs['PDE_L2'] = {k:[v.cpu().numpy()] for k,v in zip(Physics.loss_dict.keys(), pde_losses)}
            else:
                for k,v in Physics.loss_dict.items():
                    model_outputs['PDE'][k].append(v)
                for i,k in enumerate(Physics.loss_dict.keys()):
                    model_outputs['PDE_L2'][k].append(pde_losses[i].cpu().numpy())
            
        
    # Infer Solutions for train and test data
    for i, batch in enumerate(train_dataloader):
        eval_batch(i=i, batch=batch, test=False)
    
    for i, batch in enumerate(test_dataloader):
        eval_batch(i=1, batch=batch, test=True)
    
    # Concatenate into single numpy arrays per metric
    for array_keys in ['Pred','Pred_L2','Re']:
        model_outputs[array_keys] = np.concatenate(model_outputs[array_keys], axis=0)
    for dict_keys in ['PDE','PDE_L2']:
        for k,v in model_outputs[dict_keys].items():
            model_outputs[dict_keys][k] = np.concatenate(v, axis=0)

    save_path = f"{config['run_info']['out_dir']}/val_dict.npy"
    np.save(save_path, model_outputs, allow_pickle=True)
    print(f'Model validation saved to {save_path}')

if __name__ == '__main__':

    if args.config_path is None:
        ckpt, config = load_model(args.ckpt_path)
    else:
        ckpt, config = legacy_load_model(args.ckpt_path, args.config_path)
    
    validate(ckpt, config)
    

