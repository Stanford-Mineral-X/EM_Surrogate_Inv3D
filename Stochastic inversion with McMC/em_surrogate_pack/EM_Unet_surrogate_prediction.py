import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

from Surrogate3D_Unet import UNet3D_Profile


def em_surrogate_prediction(
    sigma_models, 
    param_path,
    in_channels=1,
    n_times=12, 
    base_ch=32, 
    depth=3, 
    dropout_p=0.2,
    norm="instance",
    learn_mix=False,
    use_grid_x=True,  # enables grid_sample
    N_x=19,
    Rx_loc=None
):

    # Setting hardware\
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    #print(f"Using device: {device}")
    
    # Noramlize Rx_location between [-1, 1] for projection
    Rx_loc = np.array(Rx_loc)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_coords = scaler.fit_transform(Rx_loc[:, 0].reshape(-1, 1)).ravel().astype(np.float32)
    x_coords = torch.as_tensor(x_coords, device=device)

    # Check dimension of input conductivity model
    sigma_models = np.array(sigma_models)    # Make sure the input conductivity model is an np.array type.
    assert sigma_models.shape[1:] == (45, 20, 45), f"Input shape mismatch: expect [N, 45, 20, 45], got {sigma_models.shape}."

    # Transfer input model from array to tensor
    sigma_models_tensor = torch.from_numpy(sigma_models).unsqueeze(1)  # Transfer array into tensor, shape in (B,1,45,20,45)
    sigma_models_set = TensorDataset(sigma_models_tensor)    # Package tensor into data set
    test_loader = DataLoader(sigma_models_set, batch_size=len(sigma_models_set), shuffle=False)    # Set data loader for input later

    # Define neural network
    model = UNet3D_Profile(
        in_channels,
        n_times,
        base_ch,
        depth,
        dropout_p,
        norm,
        learn_mix, 
        use_grid_x, 
        N_x,
    ).to(device) 

    # Load trained parameters and weights to the neural network
    state_dict = torch.load(param_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    # Prediction for EM data
    model.eval()
    with torch.inference_mode():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs, x_coords_norm=x_coords)    # Forward pass to get prediction

        del inputs    # Delete input to save RAM

    # Fatch predicted EM data and make it array
    outputs_flat = outputs.detach().cpu().numpy().reshape((outputs.shape[0], n_times*19))

    return outputs_flat
    
        
     