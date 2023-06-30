# Copyright (C) <2023> Xingran Liao
# @ City University of Hong Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and
# associated documentation files (the "code"), to deal in the code without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code,
# and to permit persons to whom the code is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the code.

# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE Xingran Liao BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE code OR THE USE OR OTHER DEALINGS IN THE code.
# --------------------------------------------------
from torchvision import transforms

def prepare_image256(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        # image = transforms.functional.resize(image,[256,256])
        image = transforms.functional.resize(image,256)
    # if resize and min(image.size)>224:
    #     image = transforms.functional.resize(image,[224,224])
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

