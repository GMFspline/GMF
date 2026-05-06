import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained(
    "QwenToken/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/",
    trust_remote_code=True,
)

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

def simple_tokenize_with_padding(texts, max_length=20):
    token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in texts]
    padded = tokenizer.pad(
        {"input_ids": token_ids},
        padding="max_length",
        max_length=max_length,
        return_tensors='pt'
    )
    return padded


class TrainclassDataset(Dataset):
    def __init__(self, in_path, knot_path, uptxt_path, curtxt_path, out_path, numclass=100, ):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.knot_path = knot_path
        self.uptxt_path = uptxt_path
        self.curtxt_path = curtxt_path
        self.label_path = out_path
        self.numclass = numclass
        self.knot = True
        self.end_prompt_builder = EndPromptBuilder()

        self.filenames = sorted(os.listdir(self.in_path))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        txtname = filename[:-2] + "txt"
        tensor_path = os.path.join(self.in_path, filename)
        label_path = os.path.join(self.label_path, filename)
        knot_label_path = os.path.join(self.knot_path, filename)
        uptxt_path = os.path.join(self.uptxt_path, txtname)
        curtxt_path = os.path.join(self.curtxt_path, txtname)

        with open(uptxt_path, "r", encoding="utf-8") as f:
            uptext = f.read()
        upencoding = simple_tokenize_with_padding(uptext, 147)

        with open(curtxt_path, "r", encoding="utf-8") as f:
            curtext = f.read()
        curencoding = simple_tokenize_with_padding(curtext, 48)

        end = 'Approximate the points using cuibc Bspline curve with 12 control points, predict the voxel of control points and the intervals of the knot vector.'
        endcoding = tokenizer(
            end,
            max_length=73,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tensor = torch.load(tensor_path)
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 50 - tensor.shape[0]))
        label = torch.load(label_path).reshape(-1)
        label = label.long()
        a = 12
        knot_abs = torch.Tensor(torch.load(knot_label_path)).reshape(-1)[:28-a]

        delta = knot_abs.clone()
        delta[4:] = knot_abs[4:] - knot_abs[3:27-a]
        knot = delta[4:25-a]

        non_zero_rows = tensor.any(dim=1)
        upto = upencoding["input_ids"].reshape(-1, 3).reshape(-1)

        patten_mask = non_zero_rows.long()

        uatten_mask = upencoding['attention_mask']
        catten_mask = curencoding['attention_mask']

        return (
            tensor, upto, curencoding["input_ids"], endcoding["input_ids"].squeeze(),
            patten_mask, uatten_mask[::3], catten_mask, endcoding['attention_mask'].squeeze(),
            label, knot
        )


class ShapeDataset(Dataset):
    def __init__(self, in_path, uptxt_path, curtxt_path, numclass=100):
        super().__init__()
        self.in_path = in_path
        self.uptxt_path = uptxt_path
        self.curtxt_path = curtxt_path
        self.tensor_paths = os.listdir(self.in_path)
        self.uptxt_paths = os.listdir(self.uptxt_path)
        self.curtxt_paths = os.listdir(self.curtxt_path)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        tensor_file_path = os.path.join(self.in_path, tensor_path)
        uptxt_path = self.uptxt_paths[idx]
        uptxt_path = os.path.join(self.uptxt_path, uptxt_path)
        with open(uptxt_path, "r", encoding="utf-8") as f:
            uptext = f.read()
        upencoding = simple_tokenize_with_padding(uptext, 147)

        curtxt_path = self.curtxt_paths[idx]
        curtxt_path = os.path.join(self.curtxt_path, curtxt_path)

        with open(curtxt_path, "r", encoding="utf-8") as f:
            curtext = f.read()

        curencoding = simple_tokenize_with_padding(curtext, 48)

        end = 'Approximate the points using cuibc Bspline curve with 12 control points, predict the voxel of control points and the intervals of the knot vector.'
        endcoding = tokenizer(
            end,
            max_length=73,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sample = torch.load(tensor_file_path)
        points_xyz = sample['points_xyz'].reshape(-1, 3)
        tensor = points_xyz - 0.5
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 50 - tensor.shape[0]))

        if 'model_dir' not in sample:
            raise KeyError(f"{tensor_file_path}: missing required field 'model_dir'")
        model_path = sample['model_dir']

        affine_inv = None
        for key in ('transform_affine_inv_4x4', 'affine_inv_4x4', 'inverse_transform_matrix'):
            if key in sample:
                affine_inv = sample[key]
                break
        if affine_inv is None:
            raise KeyError(
                f"{tensor_file_path}: missing inverse affine transform field; "
                "expected one of transform_affine_inv_4x4, affine_inv_4x4, inverse_transform_matrix"
            )
        affine_inv = torch.as_tensor(affine_inv, dtype=torch.float32)

        non_zero_rows = tensor.any(dim=1)
        upto = upencoding["input_ids"]

        patten_mask = non_zero_rows.long()
        uatten_mask = upencoding['attention_mask']
        catten_mask = curencoding['attention_mask']
        # Matches mamllm.val 10-tuple unpack: voxel=model_path, output=affine_inv
        return tensor, upto, curencoding["input_ids"], endcoding[
            "input_ids"].squeeze(), patten_mask, uatten_mask[::3], catten_mask, endcoding[
            'attention_mask'].squeeze(), model_path, affine_inv
