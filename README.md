# FashionCNN

Demos for training different CNNs on the Fashion-MNIST dataset.

> [!NOTE]
>
> The code in this repo was written when I was a beginner to ML (May 2024). **Do not use them in production.**
>
> However I hope this can help you learn to use CNN with PyTorch :D
>
> For explaination please refer to [d2l.ai](https://d2l.ai/chapter_convolutional-modern/index.html) (pretty good book, lots of my code written around that) :happy:

## Networks

Below is a table of all the NNs used in this demo. Specially, "MyNet" is a *truly* simple CNN designed by me for the task (not even BN!) and AlexNet (Tiny) is the simplified version of AlexNet to take exactly 28x28 inputs. The speed is measured on a single RTX 4090.

> [!NOTE]
>
> Some nets originally designed for ImageNet has to take an input size of 224 (while input size of Fashion-MNIST is 28). **However, It is not a good idea in practice to resize image inputs.**

|      Name      | Parameters | Resize (Pixels) | Speed (Examples/s) |
| :------------: | :--------: | :-------------: | :----------------: |
|     MyNet      |    0.9M    |       28        |      33171.1       |
|     LeNet      |    0.1M    |       28        |      31055.0       |
| AlexNet (Tiny) |    3.7M    |       28        |      29499.9       |
|    AlexNet     |   46.8M    |       224       |       7251.7       |
|     VGG11      |   128.8M   |       224       |       2106.5       |
|      NiN       |    2.0M    |       224       |       6880.2       |
|   GoogLeNet    |    6.0M    |       96        |      10574.0       |
|     ResNet     |   11.2M    |       96        |      16537.2       |
|    DenseNet    |    0.8M    |       96        |      16128.3       |

## Results

All nets are trained with:

- epochs = 100
- batch_size = 128
- optimizer = Ranger21, learning_rate = 0.01

|      Name      | loss_train | acc_train | acc_test  | loss_test (for fun) |
| :------------: | :--------: | :-------: | :-------: | :-----------------: |
|     MyNet      |   0.002    |   0.999   |   0.920   |         9.2         |
|     LeNet      |   0.002    | **1.000** |   0.890   |       **0.9**       |
| AlexNet (Tiny) |   0.455    |   0.998   |   0.916   |      373825.5       |
|    AlexNet     |   0.253    |   0.907   |   0.857   |      1832086.1      |
|     VGG11      |   0.001    | **1.000** |   0.923   |         4.0         |
|      NiN       |   0.058    |   0.984   |   0.922   |       4089.5        |
|   GoogLeNet    |   0.003    | **1.000** | **0.943** |         5.1         |
|     ResNet     |   0.002    | **1.000** |   0.927   |        16.7         |
|    DenseNet    | **0.000**  | **1.000** |   0.942   |        13.1         |

> [!IMPORTANT]
>
> As most of the models were designed for ImageNet challenges, the tests on Fashion-MNIST maybe somehow unfair.
>
> **ONLY FOR LEARNING PURPOSES**
