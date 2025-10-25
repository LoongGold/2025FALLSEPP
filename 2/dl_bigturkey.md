```python
!wget https://raw.githubusercontent.com/Atcold/NYU-DLSP21/refs/heads/master/res/plot_lib.py
```

    --2025-10-11 15:53:23--  https://raw.githubusercontent.com/Atcold/NYU-DLSP21/refs/heads/master/res/plot_lib.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4605 (4.5K) [text/plain]
    Saving to: ‘plot_lib.py.2’
    
    plot_lib.py.2         0%[                    ]       0  --.-KB/s               plot_lib.py.2       100%[===================>]   4.50K  --.-KB/s    in 0s      
    
    2025-10-11 15:53:24 (71.0 MB/s) - ‘plot_lib.py.2’ saved [4605/4605]
    
    


```python
import random
import torch
from torch import nn,optim
import math
from IPython import display
from plot_lib import plot_data, plot_model, set_default

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)
seed = 12345
random.seed(seed)

torch.manual_seed(seed)
N=1000
D=2
C=3
H=100
```

    device:  cuda:0
    


```python
X=torch.zeros(N*C, D).to(device)
Y=torch.zeros(N*C,dtype=torch.long).to(device)
for c in range(C):
    index=0
    t= torch.linspace(0,1,N)
    inner_var = torch.linspace((2*math.pi/C)*c,(2*math.pi/C)*(2+c),N)+ torch.randn(N) * 0.2
    for ix in range(N * c,N *(c + 1)):
        X[ix]= t[index] * torch.FloatTensor([math.sin(inner_var[index]),math.cos(inner_var[index])])
        Y[ix]=c
        index += 1
print("shapes:")
print("X:", X.size())
print("Y:", Y.size())
```

    shapes:
    X: torch.Size([3000, 2])
    Y: torch.Size([3000])
    


```python
plot_data(X,Y)
```




    <matplotlib.collections.PathCollection at 0x7e8090c160c0>




    
![png](dl_bigturkey_files/dl_bigturkey_3_1.png)
    

