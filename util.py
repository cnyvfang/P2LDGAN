from pytorch_fid import fid_score, inception
import torch
import numpy as np

def device():
    return torch.device('1')

def test_calculate_fid_given_statistics(mocker, tmp_path, device):
    dim = 2048
    m1, m2 = np.zeros((dim,)), np.ones((dim,))
    sigma = np.eye(dim)

    def dummy_statistics(path, model, batch_size, dims, device, num_workers):
        if path.endswith('1'):
            return m1, sigma
        elif path.endswith('2'):
            return m2, sigma
        else:
            raise ValueError

    mocker.patch('pytorch_fid.fid_score.compute_statistics_of_path',
                 side_effect=dummy_statistics)

    dir_names = ['1', '2']
    paths = []
    for name in dir_names:
        path = tmp_path / name
        path.mkdir()
        paths.append(str(path))

    fid_value = fid_score.calculate_fid_given_paths(paths,
                                                    batch_size=dim,
                                                    device=device,
                                                    dims=dim,
                                                    num_workers=0)

    # Given equal covariance, FID is just the squared norm of difference
    assert fid_value == np.sum((m1 - m2)**2)