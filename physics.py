import torch

def ff(x, y):
    return 2 * x.pow(2) + 1


def f_norm(F, u):
    return F * u.norm(dim=-1, keepdim=True)

def finite_diff_grad_godunov_eikonal(output_grid, dx=1.0, dy=1.0, eps=1e-12):
    """
    For steady eikonal |grad u| = rhs with u increasing away from source.
    """
    u = output_grid

    Dx_minus = torch.zeros_like(u)
    Dx_plus  = torch.zeros_like(u)
    Dy_minus = torch.zeros_like(u)
    Dy_plus  = torch.zeros_like(u)

    Dx_minus[:, 1:, :] = (u[:, 1:, :] - u[:, :-1, :]) / dx
    Dx_minus[:, 0, :]  = Dx_minus[:, 1, :]

    Dx_plus[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / dx
    Dx_plus[:, -1, :]  = Dx_plus[:, -2, :]

    Dy_minus[:, :, 1:] = (u[:, :, 1:] - u[:, :, :-1]) / dy
    Dy_minus[:, :, 0]  = Dy_minus[:, :, 1]

    Dy_plus[:, :, :-1] = (u[:, :, 1:] - u[:, :, :-1]) / dy
    Dy_plus[:, :, -1]  = Dy_plus[:, :, -2]
    
    ux = torch.maximum(
        torch.clamp(Dx_minus, min=0.0),
        torch.clamp(-Dx_plus, min=0.0)
    )

    uy = torch.maximum(
        torch.clamp(Dy_minus, min=0.0),
        torch.clamp(-Dy_plus, min=0.0)
    )

    grad_norm = torch.sqrt(ux**2 + uy**2 + eps)

    return ux, uy, grad_norm

def eikonal_residual(output, FF, dx, dy):
    _,_,output_grad_norm = finite_diff_grad_godunov_eikonal(output, dx, dy)
    output_grad_norm = FF * output_grad_norm # (B, Nx, Ny)
    
    return output_grad_norm - 1
        