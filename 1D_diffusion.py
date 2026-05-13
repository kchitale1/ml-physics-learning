import torch
from torch.utils.data import Dataset, DataLoader

NU = 0.01  # diffusivity

class DiffusionDataset(Dataset):
    def __init__(self, n_samples, x_range=(0, 1), t_range=(0, 1), nu=NU):
        # store n_samples, ranges, nu
        # pre-sample (x, t) pairs randomly and store them
        # hint: torch.rand gives you uniform samples
        self.x=torch.rand(n_samples) * (x_range[1] - x_range[0]) + x_range[0]
        self.t=torch.rand(n_samples) * (t_range[1] - t_range[0]) + t_range[0]
        self.nu=nu

    def __len__(self):
        return len(self.x)  # or len(self.t), they should be the same

    def __getitem__(self, idx):
        # return (input_tensor, target_tensor) for sample idx
        # input: [x, t], output: [u(x,t)]
        x = self.x[idx]
        t = self.t[idx]
        u = torch.exp(-torch.pi**2 * self.nu * t) * torch.sin(torch.pi * x)
        return torch.stack([x, t]), u.unsqueeze(0)  # cleaner

def my_collate(batch):
    # batch is a list of (input_tensor, target_tensor) tuples
    # your job: stack them manually, and print the batch size as a sanity check
    inputs, targets = zip(*batch)
    print(f"Collating batch of {len(inputs)} samples")
    return torch.stack(inputs), torch.stack(targets)

# after the class: instantiate, wrap in DataLoader, iterate one batch

if __name__ == '__main__':
    dataset = DiffusionDataset(n_samples=1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=2, collate_fn=my_collate)
    for x, y in loader:
        print(x.shape, y.shape)
        break
