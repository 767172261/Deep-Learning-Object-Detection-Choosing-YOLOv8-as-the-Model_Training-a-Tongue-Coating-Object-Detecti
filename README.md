舌苔目标检测数据集
map50精度99%以上
2460张图像，6类
["黑色舌苔", "白霉舌苔", "紫色舌苔", "红色舌苔黄厚腻", "红色舌苔厚腻", "白色舌苔厚腻"]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bd45212ebf214ab89316831da3167d0c.png)

["black tongue coating", "map tongue coating", "purple tongue coating","red tongue yellow fur thick greasy fur","The red tongue is thick and greasy","The white tongue is thick and greasy"]

数据集划分:训练集、验证集、测试集【1968:246:246】
YOLO标注

选择YOLOv8作为模型，因为它在目标检测任务中表现出色，并且支持YOLO格式的数据标注。，包括如何准备数据、训练模型以及评估其性能。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61affffa9e0043cf8735d24a2238bc33.png)

### 1. 环境准备

首先，确保安装了必要的库和工具：

```bash
pip install ultralytics opencv-python-headless matplotlib
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0343d36d49f6400396eeb2b335d2bf6c.png)

### 2. 数据集准备

假设您的数据集已经按照YOLO格式进行标注，结构如下：

```
tongue_coating_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   └── ...
│   └── test/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    ├── val/
    │   ├── img1.txt
    │   └── ...
    └── test/
        ├── img1.txt
        └── ...
data_tongue_coating.yaml
```

`data_tongue_coating.yaml` 文件内容示例：

```yaml
train: ./tongue_coating_dataset/images/train/
val: ./tongue_coating_dataset/images/val/

nc: 6  # 类别数量
names: ['black tongue coating', 'map tongue coating', 'purple tongue coating', 'red tongue yellow fur thick greasy fur', 'The red tongue is thick and greasy', 'The white tongue is thick and greasy']
```

### 3. 训练模型

接下来，我们将使用YOLOv8来训练我们的模型。

#### 训练脚本

```python
from ultralytics import YOLO
import os

def main_train():
    # 加载预训练的YOLOv8n模型或从头开始定义模型
    model = YOLO('yolov8n.yaml')  # 或者直接加载预训练权重，如 'yolov8n.pt'
    
    results = model.train(
        data='./data_tongue_coating.yaml',
        epochs=200,  # 根据需要调整
        imgsz=640,
        batch=16,
        project='./runs/detect',
        name='tongue_coating_detection',
        optimizer='Adam',  # 可以根据需要选择优化器
        device='0',  # 使用GPU编号
        save=True,
        cache=True,
        verbose=True,
        patience=10,  # 提前停止的耐心值
        lr0=0.01,  # 初始学习率
        lrf=0.1,  # 最终学习率（相对于lr0的比例）
    )

if __name__ == '__main__':
    main_train()
```

### 4. 模型评估

训练完成后，我们需要对模型进行评估，查看其在验证集上的表现。

#### 评估脚本

```python
from ultralytics import YOLO

def evaluate_model():
    model = YOLO('./runs/detect/tongue_coating_detection/weights/best.pt')
    
    metrics = model.val(data='./data_tongue_coating.yaml', imgsz=640, batch=16)
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == '__main__':
    evaluate_model()
```

### 5. 测试与可视化

我们可以使用训练好的模型对测试集中的图像进行预测，并将结果可视化。

#### 测试与可视化脚本

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def detect_and_visualize(image_path):
    model = YOLO('./runs/detect/tongue_coating_detection/weights/best.pt')
    
    results = model.predict(source=image_path, save=True)  # 设置save为True保存预测结果
    
    for result in results:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy.tolist() + [box.conf.item(), box.cls.item()]
            label = model.names[int(cls)]
            if conf > 0.5:  # 设置置信度阈值
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    detect_and_visualize('path/to/test/image.jpg')
```

### 总结

构建一个完整的基于YOLOv8的舌苔目标检测系统。从数据准备、模型训练、模型评估以及结果的可视化展示。


