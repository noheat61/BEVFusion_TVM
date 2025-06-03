import torch
import torch.nn as nn
import torch.nn.functional as F

def bev_pool(geom_feats, x, bevpool_layer):

    def gen_dx_bx(xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).to(x.device)
        bx = torch.Tensor(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).to(x.device)
        nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                            for row in [xbound, ybound, zbound]]).to(x.device)
        return dx, bx, nx
    
    xbound=tuple([-54.0, 54.0, 0.3])
    ybound=tuple([-54.0, 54.0, 0.3])
    zbound=tuple([-10.0, 10.0, 20.0])
    
    dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
    dx = nn.Parameter(dx, requires_grad=False)
    bx = nn.Parameter(bx, requires_grad=False)
    nx = nn.Parameter(nx, requires_grad=False)

    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # flatten x
    x = x.reshape(Nprime, C)

    # flatten indices
    geom_feats = ((geom_feats - (bx - dx / 2.0)) /
                    dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat([
        torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
        for ix in range(B)
    ])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    # filter out points that are outside box
    kept = ((geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < nx[2]))
    x = x[kept]
    geom_feats = geom_feats[kept]

    x = bevpool_layer(x, geom_feats, B, nx[2], nx[0], nx[1])

    # collapse Z
    final = torch.cat(x.unbind(dim=2), 1)

    return final

def voxelize(points, voxel_layer, voxelize_reduce=True):
    feats, coords, sizes = [], [], []
    for k, res in enumerate(points):
        ret = voxel_layer(res)
        if len(ret) == 3:
            f, c, n = ret
        else:
            assert len(ret) == 2
            f, c = ret
            n = None
        feats.append(f)
        coords.append(F.pad(c, (1, 0), mode='constant', value=k))
        if n is not None:
            sizes.append(n)

    feats = torch.cat(feats, dim=0)
    coords = torch.cat(coords, dim=0)
    if len(sizes) > 0:
        sizes = torch.cat(sizes, dim=0)
        if voxelize_reduce:
            feats = feats.sum(
                dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

    return feats, coords, sizes
