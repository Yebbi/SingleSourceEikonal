import torch
import torch.nn as nn
import torch.nn.functional as F

class branch_net(nn.Module):
    def __init__(self, hidden_dim):
        super(branch_net,self).__init__()

        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, x):
        """
        x:
            (B,2)
            source point

        output:
            (B,p)
        """

        return self.net(x)
    
class trunk_net(nn.Module):

    def __init__(self, hidden_dims, act):

        super(trunk_net,self).__init__()

        layers=[]

        for i in range(len(hidden_dims)-1):
            layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )

        self.layers=nn.ModuleList(layers)

        self.act=act


    def forward(self,x):

        x=self.act(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x=self.act(layer(x))

        x=self.layers[-1](x)

        return x
            
class deepOnet(nn.Module):

    def __init__(
        self,
        branch,
        trunk_u,
        Nx,
        Ny
    ):

        super(deepOnet,self).__init__()

        self.branch = branch
        self.trunk_u = trunk_u

        self.hidden_dim = branch.hidden_dim

        self.Nx=Nx
        self.Ny=Ny



    def forward(
        self,
        source,
        xyF,
        distance,
    ):

        """
        source: (B,2)

        xyF: (Nx*Ny,4)

        return: d_A(x,source) (B,Nx,Ny)
        """

        B = source.shape[0]
     
        ## branch
        branch_out = self.branch(source)# (B,p)

        ## trunk
        trunk_out = self.trunk_u(xyF) # (Nx*Ny,p)

        # expand only trunk feature
        trunk_out = trunk_out.unsqueeze(0).expand(B,-1,-1) # (B,Nx*Ny,p)

        # ------------------
        # dot product
        # ------------------
        out = torch.einsum("bp,bmp->bm",branch_out,trunk_out)
        out = out*distance
        out = out.view(B,self.Nx,self.Ny)

        return out
