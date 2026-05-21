import torch
import torch.optim.lr_scheduler as schedulers
import numpy as np
from pathlib import Path
import sys
import os
import yaml
import argparse

from src import OpenFoam_Mesh
from src import Physics_loss_controller
from .utils import LpLoss, get_GNOT_model, get_dataset_loaders, create_save_dir
from .validation import validate

# load configuration from json
parser = argparse.ArgumentParser("TRAIN THE FLASH GNOT TRANSFORMER")
parser.add_argument('--config', type=str, help="yaml configuration file")
args = parser.parse_args()

if __name__ == '__main__':

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    params = config['parameters']
    dataset_params = config['dataset_settings']

    '''
    Set seeds for random numbers
    NOTE This is very important for training convergence. AD-FVM offers a field-gradient loss calculation for training.
         This script only demonstrates how it is used and does not intend to improve physics-informed training.
         Different random numbers (as used in model initialization) may not reproduce results similar to the paper.
    '''
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    # Prepare Dataset
    train_dataloader, test_dataloader = get_dataset_loaders(params, dataset_params)

    # Setup FVM and Physics-Loss Controller
    Mesh = OpenFoam_Mesh(openfoam_case_dir=Path(dataset_params['openFOAM_case_dir']), dtype=torch.float32, corrected=True)
    Mesh._print_mesh_components()

    # Setup Navier-Stokes loss calculator
    Physics = Physics_loss_controller(physics_fn='ns_dns_incompress', 
                                      loss_fn=LpLoss(), 
                                      channel_dict={'U':[0,1], 'p':[2]}, # 2D mesh
                                      mesh=Mesh)
    
    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_GNOT_model(dataset_params=dataset_params, params=params).to(device)

    # Move FVM framework to device
    Mesh.to(device)

    # Setup Training Utilities
    data_weighting, pde_weighting = params.get('data_weighting',1), params.get('pde_loss_weight',0)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])
    scheduler = schedulers.OneCycleLR(optimizer, max_lr=params['init_lr'], total_steps=params['epochs'],
                                      **{k: params[k] for k in ['div_factor', 'pct_start', 'final_div_factor'] if k in params}
                                      )
    Loss_fn = LpLoss()
    
    # Training Loop
    for epoch in range(params['epochs']):
        train_cumu_loss = 0
        data_cumu_loss = 0
        for batch in train_dataloader:
            x, input_f, y = batch
            x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)
            optimizer.zero_grad()

            out = model(x=x, inputs=input_f)
            data_loss = Loss_fn(out,y)

            pde_losses = Physics.compute(out.unsqueeze(1), Re=input_f[0], volume_weighted=True)
            pde_loss = torch.stack(pde_losses).mean()
            
            training_loss = pde_loss * pde_weighting + data_loss * data_weighting
            
            train_cumu_loss += training_loss.item()
            data_cumu_loss += data_loss.item()

            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
            optimizer.step()
            
        with torch.no_grad():
            test_cumu_loss = 0
            for batch in test_dataloader:
                x, input_f, y = batch
                x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)
                out = model(x=x, inputs=input_f)
                data_loss = Loss_fn(out,y)
                test_cumu_loss += data_loss.item()
        
        train_cumu_loss /= len(train_dataloader)
        data_cumu_loss /= len(train_dataloader)
        test_cumu_loss /= len(test_dataloader)
        print(f"{epoch:6}/{params['epochs']} | Train Loss: {train_cumu_loss:10.4f} | Train Data Loss: {data_cumu_loss:7.4f} | Test Data Loss: {test_cumu_loss:7.4f}")

    print('Training Complete')

    save_dir = create_save_dir(config)

    # Save model and run outcome
    config['run_info']['train_loss'] = train_cumu_loss
    config['run_info']['train_data_loss'] = data_cumu_loss
    config['run_info']['test_data_loss'] = test_cumu_loss

    checkpoint = {'epoch': epoch, 
                  'model_state_dict': model.state_dict(), 
                  'optimizer_state_dict': optimizer.state_dict(), 
                  'config': config}
    full_save_path = f"{config['run_info']['out_dir']}/checkpoint_epoch_{epoch+1}.pth"
    torch.save(checkpoint, full_save_path)
    print(f'Model saved to: {full_save_path}')

    # Validate
    validate(model.state_dict(), config)