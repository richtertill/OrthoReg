import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
import os
from orthoreg.paths import DATA_DIR
from orthoreg.data.datasets import init_dataloaders
import numpy as np

# Lightning Datamodule
class HybridDataModule(L.LightningDataModule):
    def __init__(self, cfg, data_dict, feature_library):
        super().__init__()
        # Store all datasets
        self.train_data = data_dict.get('train_data', None)
        self.id_test_data = data_dict.get('id_test_data', None)
        self.ood_test_data_t2 = data_dict.get('ood_test_data_t2', None)
        self.ood_test_data_t3 = data_dict.get('ood_test_data_t3', None)
        # Store extra datasets
        self.train_data_extra = data_dict.get('train_data_extra', None)
        self.id_test_data_extra = data_dict.get('id_test_data_extra', None)
        self.ood_test_data_t2_extra = data_dict.get('ood_test_data_t2_extra', None)
        self.ood_test_data_t3_extra = data_dict.get('ood_test_data_t3_extra', None)
        
        # Store the dictionary itself for later access
        self.data_dict = data_dict
        
        self.feature_library = feature_library
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.W_true = data_dict.get('W_true', None)
        self.cfg = cfg
        
        # Initialize all datasets as None
        self.train_dataset = None
        self.id_test_dataset = None
        self.ood_test_dataset_t2 = None
        self.ood_test_dataset_t3 = None
        self.train_dataset_extra = None
        self.id_test_dataset_extra = None
        self.ood_test_dataset_t2_extra = None
        self.ood_test_dataset_t3_extra = None
        
        # Add this property for hyperparameter logging
        self._log_hyperparams = True
    
    def _prepare_dataset(self, data):
        """Helper method to prepare a single dataset"""
        # Handle None data
        if data is None:
            return None
            
        # Handle tuple data structure
        if isinstance(data, tuple):
            # Extract both parts of the tuple (task_id and data)
            task_id, data = data[0]
        
        y = data.y
        # Transform each trajectory independently
        transformed_y_list = []
        for traj in y:
            # Convert to numpy array if it's a tensor
            if isinstance(traj, torch.Tensor):
                traj = traj.cpu().numpy()
            # Transform and append
            transformed = self.feature_library.fit_transform(traj.reshape(-1, traj.shape[-1]))
            transformed_y_list.append(transformed)
        
        # Manually stack the arrays
        transformed_y = np.array(transformed_y_list)
        # Convert to tensor
        transformed_y = torch.tensor(transformed_y, dtype=torch.float32, device=self.device)
        
        t = data.t
        dy = data.dy[self.cfg.model.diff_method].to(self.device)
        
        # Expand t to match batch dimension [n_tasks, n_timepoints]
        t_expanded = t.unsqueeze(0).expand(y.shape[0], -1)
        
        return TensorDataset(y, transformed_y, dy, t_expanded)

    def setup(self, stage=None):
        # Use a simple and reliable device detection method
        # The trainer device might not be available during setup, so we'll use a fallback
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"HybridDataModule setup - device: {self.device}")
        
        # Prepare all datasets
        self.train_dataset = self._prepare_dataset(self.train_data)
        self.id_test_dataset = self._prepare_dataset(self.id_test_data)
        self.ood_test_dataset_t2 = self._prepare_dataset(self.ood_test_data_t2)
        self.ood_test_dataset_t3 = self._prepare_dataset(self.ood_test_data_t3)
        
        # Store W_true as a class attribute instead, if it exists
        if self.W_true is not None:
            self.W_true_tensor = torch.stack([self.W_true[key] for key in self.W_true.keys()])
        else:
            self.W_true_tensor = None
        
        # Prepare extra datasets
        self.train_dataset_extra = self._prepare_dataset(self.train_data_extra)
        self.id_test_dataset_extra = self._prepare_dataset(self.id_test_data_extra)
        self.ood_test_dataset_t2_extra = self._prepare_dataset(self.ood_test_data_t2_extra)
        self.ood_test_dataset_t3_extra = self._prepare_dataset(self.ood_test_data_t3_extra)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.training.batch_size, shuffle=False)

    def test_dataloader(self):
        # Return a list of test dataloaders, skipping None datasets
        datasets = [
            self.id_test_dataset,
            self.ood_test_dataset_t2,
            self.ood_test_dataset_t3,
            self.id_test_dataset_extra,
            self.ood_test_dataset_t2_extra,
            self.ood_test_dataset_t3_extra
        ]
        test_loaders = []
        for dataset in datasets:
            if dataset is not None:
                test_loaders.append(DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=False))
        return test_loaders
    

def setup_dataloaders(dataset_name, cfg, feature_library):
    (train_data, train_data_extra), \
    (id_test_data, id_test_data_extra), \
    (ood_test_data_t2, ood_test_data_t2_extra), \
    (ood_test_data_t3, ood_test_data_t3_extra), \
    W_true = init_dataloaders(
        dataset=dataset_name,
        feature_library=feature_library,
        n_samples=cfg.dataset.forecaster.n_samples,
        times=cfg.dataset.forecaster.times,
        granularity=cfg.dataset.forecaster.granularity,
        sampling_scheme=cfg.dataset.forecaster.sampling_scheme,
        buffer_filepath=os.path.join(DATA_DIR, dataset_name),
        cfg=cfg
    )

    # --- Compute W_true for relevant datasets --- 
    # Fit the feature library on training data (required by compute_true_W)
    # Ensure y and t are tensors for fitting if necessary
    if not isinstance(train_data.y, torch.Tensor):
        train_y = torch.tensor(train_data.y, dtype=torch.float32)
    else:
        train_y = train_data.y
    if not isinstance(train_data.t, torch.Tensor):
        train_t = torch.tensor(train_data.t, dtype=torch.float32)
    else:
        train_t = train_data.t
        
    # PySINDy.fit expects numpy arrays.
    feature_library.fit(train_y.numpy(), train_t.numpy())

    # Compute W_true for base datasets
    print("Computing W_true for datasets...")
    train_data.compute_true_W(feature_library, train_data)
    id_test_data.compute_true_W(feature_library, train_data)
    ood_test_data_t2.compute_true_W(feature_library, train_data)
    ood_test_data_t3.compute_true_W(feature_library, train_data)
    print("Finished computing W_true.")

    # Copy W_true to extrapolation datasets (assuming parameters are the same)
    if hasattr(train_data, 'W_true'):
        train_data_extra.W_true = train_data.W_true
    if hasattr(id_test_data, 'W_true'):
        id_test_data_extra.W_true = id_test_data.W_true
    if hasattr(ood_test_data_t2, 'W_true'):
        ood_test_data_t2_extra.W_true = ood_test_data_t2.W_true
    if hasattr(ood_test_data_t3, 'W_true'):
        ood_test_data_t3_extra.W_true = ood_test_data_t3.W_true
    # --- End Compute W_true ---

    data_dict = {
        'train_data': train_data,
        'train_data_extra': train_data_extra,
        'id_test_data': id_test_data,
        'id_test_data_extra': id_test_data_extra,
        'ood_test_data_t2': ood_test_data_t2,
        'ood_test_data_t2_extra': ood_test_data_t2_extra,
        'ood_test_data_t3': ood_test_data_t3,
        'ood_test_data_t3_extra': ood_test_data_t3_extra,
        'W_true': W_true
    }
    return data_dict