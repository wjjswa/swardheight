import torch.nn as nn
import ops
import utils
import torch
import cv2 as cv
import numpy as np

def calculate_metrics(preds, targets):
    preds = preds.cpu().float()
    targets = targets.cpu().float()
    n = targets.size(0)
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else 0
    percentage_error = torch.abs((targets - preds) / targets)
    mape = torch.mean(percentage_error).item() * 100
    return mae,rmse, r2, mape


class PatchEmbeddings(nn.Module):
    """Extracts patch embeddings from input images."""
    def __init__(self, C_in, C_cur, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_cur,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
    def forward(self, x):
        x = self.projection(x)  # (B, C_cur, H/P, W/P)
        x = x.flatten(2)  # Flatten the spatial dimensions (H/P, W/P) into one (B, C_cur, N)
        x = x.transpose(1, 2)  # Transpose to (B, N, C_cur)
        return x
class FixedCell(nn.Module):
    """ Fixed cell based on genotype """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.reduction_p = reduction_p

        # Handle the reduction preprocessing
        if reduction_p:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        else:
            self.preproc0 = nn.Identity()

        if reduction:
            self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)
        else:
            self.preproc1 = nn.Identity()

        # Build dag based on genotype
        self.dag = nn.ModuleList()
        for i in range(len(genotype)):
            self.dag.append(nn.ModuleList())
            node_ops = genotype[i]
            for j in range(2 + i):
                for op_name, node_index in node_ops:
                    stride = 1
                    op = ops.OPS[op_name](C, stride,affine=False) if reduction else ops.tf_OPS[op_name](C, stride,affine=False)
                    if node_index == j:
                        self.dag[i].append(op)

    def forward(self, s0, s1):
        if not self.reduction_p and self.reduction:
            batch_size, _, C = s0.shape
            s0 = s0.reshape(batch_size, C, 14, 14)
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]

        for edges in self.dag:
            s_cur = sum(edges[i](s) for i, s in enumerate(states))
            states.append(s_cur)

        if not self.reduction_p and not self.reduction:
            s_out = torch.cat(states[2:], dim=2)
            batch_size, _, C = s_out.shape
            s_out = s_out.view(batch_size, C, 14, 14)
        else:
            s_out = torch.cat(states[2:], dim=1)

        return s_out


class FixedCNN(nn.Module):
    def __init__(self, genotype, C_in, C, n_classes, n_layers, n_nodes, stem_multiplier=3):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = PatchEmbeddings(C_in, C, 16)

        C_pp, C_p, C_cur = C, C, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):

            reduction = True if i in [n_layers // 3, 2 * n_layers // 3] else False
            cell = FixedCell(genotype.normal if not reduction else genotype.reduce,
                             C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_pp, C_p = C_p, C_cur * n_nodes

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits

def pad_array(array, target_size,no_data_value):
    original_size = array.shape
    pad_rows = max(0, target_size[1] - original_size[1])
    pad_cols = max(0, target_size[2] - original_size[2])
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left
    pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=no_data_value)
    return padded_array

def load_single_file(file_path,min_val,max_val,device):
    image = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found or the format is not supported")
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        target_size = (1, 224, 224)
    elif image.ndim == 3:
        image = np.transpose(image, (2, 0, 1))  # Reorder to C x H x W
        target_size = (3, 224, 224)
    else:
        raise ValueError("Image dimensions are unusual, check the image format")

    ratio = min(target_size[1] / image.shape[1], target_size[2] / image.shape[2])
    if ratio < 1:
        if image.shape[0] == 1:
            image = np.transpose(image, (1, 2, 0))
            image = cv.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
            image = np.expand_dims(image, axis=0)
        else:
            image = np.transpose(image, (1, 2, 0))
            image = cv.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
            image = np.transpose(image, (2, 0, 1))

    padded_array = pad_array(image, target_size, 0)


    # Convert to float32 for normalization
    x = padded_array.astype(np.float32)

    # Normalize
    if min_val is not None and max_val is not None:
        denom = (max_val - min_val)
        if denom == 0:
            raise ValueError("max_val equals min_val; cannot normalize.")
        x = (x - float(min_val)) / float(denom)
    else:
        # Automatic 0–1 scaling
        if x.dtype == np.float32 or x.dtype == np.float64:
            m = np.nanmax(x)
            if m > 1.0:
                x = x / (m if m > 0 else 1.0)
        else:
            if image.dtype == np.uint8:
                x = x / 255.0
            elif image.dtype == np.uint16:
                x = x / 65535.0
            else:
                m = np.nanmax(x)
                x = x / (m if m > 0 else 1.0)

    x = np.clip(x, 0.0, 1.0)
    tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # shape [1, C, H, W]
    return tensor

def prediction(X, model, min_val, max_val, device):
    scale_diff = max_val - min_val
    model = model.to(device)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        logits_rescaled = logits * scale_diff + min_val

    return logits_rescaled.cpu()


def test_model_train_iterate(genotype,img_tensor,min_val,max_val,in_channels,fc_output,device):
    genotype = utils.load_genotype(genotype)
    test_model = FixedCNN(genotype, C_in=in_channels, C=16, n_classes=fc_output,
                          n_layers=3,
                          n_nodes=2, )
    test_model.load_state_dict(torch.load("best_sward_model.pth", map_location=device))
    test_model.to(device)
    test_preds = prediction(img_tensor, test_model, min_val, max_val, device)
    return test_preds

def main_grid():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    genotype = 'best_sward_pt.json'
    image_path= 'data/grasslab/map123only/dsm/map123_together_real_dsm_mean/map1_1dsm_segment_1.jpg'

    # obtain dataset parameters
    # dataset_dir= 'data/grasslab/map123only/dsm/map123_together_real_dsm_mean'
    # min_val,max_val,in_channels,fc_output = (utils.load_data(dataset_dir))
    min_val= 13.444444444444445
    max_val= 30.88888888888889
    in_channels= 1
    fc_output= 1

    img_tensor = load_single_file(image_path, min_val, max_val,device=device)

    test_preds = test_model_train_iterate(genotype, img_tensor, min_val,max_val,in_channels,fc_output,device)
    print(f"test_preds: {test_preds}")

if __name__ == "__main__":
    main_grid()