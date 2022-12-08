import torch
import kornia

def get_GD_kernel_1() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [1., 0., 0.],
        [0., -1., 0.],
        [0., 0., 0.],
    ]).to(torch.float32)
def get_GD_kernel_2() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 0.],

    ]).to(torch.float32)
def get_GD_kernel_3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 1.],
        [0., -1., 0.],
        [0., 0., 0.],
    ]).to(torch.float32)
def get_GD_kernel_4() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 0.],
        [0., -1., 1.],
        [0., 0., 0.],
    ])
def get_GD_kernel_5() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 0.],
        [0., -1.,0.],
        [0., 0., 1.],
    ]).to(torch.float32)
def get_GD_kernel_6() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 0.],
        [0., -1., 0.],
        [0., 1., 0.],
    ]).to(torch.float32)
def get_GD_kernel_7() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 0.],
        [0., -1., 0.],
        [1., 0., 0.],
    ]).to(torch.float32)
def get_GD_kernel_8() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0., 0., 0.],
        [1., -1., 0.],
        [0., 0., 0.],
    ]).to(torch.float32)


def GD_blur(input: torch.Tensor,
             border_type: str = 'reflect') -> torch.Tensor:
    kernel1: torch.Tensor = get_GD_kernel_1().unsqueeze(0)
    kernel2: torch.Tensor = get_GD_kernel_2().unsqueeze(0)
    kernel3: torch.Tensor = get_GD_kernel_3().unsqueeze(0)
    kernel4: torch.Tensor = get_GD_kernel_4().unsqueeze(0)
    kernel5: torch.Tensor = get_GD_kernel_5().unsqueeze(0)
    kernel6: torch.Tensor = get_GD_kernel_6().unsqueeze(0)
    kernel7: torch.Tensor = get_GD_kernel_7().unsqueeze(0)
    kernel8: torch.Tensor = get_GD_kernel_8().unsqueeze(0)
    k1=abs(kornia.filter2D(input, kernel1, border_type))
    k2=abs(kornia.filter2D(input, kernel2, border_type))
    k3=abs(kornia.filter2D(input, kernel3, border_type))
    k4=abs(kornia.filter2D(input, kernel4, border_type))
    k5=abs(kornia.filter2D(input, kernel5, border_type))
    k6=abs(kornia.filter2D(input, kernel6, border_type))
    k7=abs(kornia.filter2D(input, kernel7, border_type))
    k8=abs(kornia.filter2D(input, kernel8, border_type))
    result=k1+k2+k3+k4+k5+k6+k7+k8
    return result

def LTP_blur(input: torch.Tensor,
             threashold,
             border_type: str = 'reflect') -> torch.Tensor:
    kList = [];
    kernel1: torch.Tensor = get_GD_kernel_1().unsqueeze(0)
    kList.append(kernel1);
    kernel2: torch.Tensor = get_GD_kernel_2().unsqueeze(0)
    kList.append(kernel2);
    kernel3: torch.Tensor = get_GD_kernel_3().unsqueeze(0)
    kList.append(kernel3);
    kernel4: torch.Tensor = get_GD_kernel_4().unsqueeze(0)
    kList.append(kernel4);
    kernel5: torch.Tensor = get_GD_kernel_5().unsqueeze(0)
    kList.append(kernel5);
    kernel6: torch.Tensor = get_GD_kernel_6().unsqueeze(0)
    kList.append(kernel6);
    kernel7: torch.Tensor = get_GD_kernel_7().unsqueeze(0)
    kList.append(kernel7);
    kernel8: torch.Tensor = get_GD_kernel_8().unsqueeze(0)
    kList.append(kernel8);
    resultList = [];
    weight = 1;
    for kernel in kList:
        k=kornia.filter2D(input, kernel, border_type)
        k=torch.where(k>threashold,torch.ones(1,1),k);
        k=torch.where(k<-threashold,-torch.ones(1,1),k);
        k=torch.where(abs(k)<1,torch.zeros(1,1),k);
        resultList.append(k*weight);
        weight *=2;
    result=resultList[0]+resultList[1]+resultList[2]+resultList[3]+resultList[4]+resultList[5]+resultList[6]+resultList[7];
    return result;
