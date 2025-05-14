import sys
sys.path.insert(0, './')

from ultralytics import YOLO
import torch
from ultralytics.nn.modules import C2f, Conv, Bottleneck, SPPF, Detect

threshold = 0

# 对bottleneck结构的两个串联Conv剪枝
# 也考虑到通用性，串联第二个组件可以不是Conv
def prune_(cv1 : Conv, cv2 : Conv):
    gamma = cv1.bn.weight.detach()
    keep_id = []
    local_threshold = threshold # 设置初始阈值

    # 由于可能存在恰好本层权重都很小基本不超过threshold，此时为了保留一定程度最少的权重需要不断下降阈值
    # 不断下降local_threshold直到权重保留数量超过8，这个8是个经验值
    while len(keep_id) < 8:
        keep_id = torch.where(gamma.abs() > local_threshold)[0]
        local_threshold = local_threshold * 0.5

    # 剪枝后保留的位置也是新的输出位置
    new_feature_num = len(keep_id)

    # weight形状为 [out_channels, in_channels, kernel_h, kernel_w]
    # 像下面gamma[keep_id]这种操作就是在out_channels维度上截取。其他维度不变
    cv1.bn.weight.data = gamma[keep_id]
    beta = cv1.bn.bias.detach()
    cv1.bn.bias.data = beta[keep_id] # 这里直接使用了weight的权重决定bias，因为wx+b中weight对x是乘性直接决定可用与否，因此与其相关性大
    cv1.bn.running_var = cv1.bn.running_var[keep_id] # 方差
    cv1.bn.running_mean = cv1.bn.running_mean[keep_id] # 均值
    cv1.bn.num_features = new_feature_num
    # 由于bn在Conv组件里直接跟在conv后，因此两者的索引是完全相同
    cv1.conv.weight.data = cv1.conv.weight.data[keep_id]
    cv1.conv.out_channels = new_feature_num # conv后跟着bn，两者feature对齐

    # conv里可能根据maxpool/unsample的存在，bias可以没有
    if not cv1.conv.bias is None:
        cv1.conv.bias = cv1.conv.bias[keep_id]

    # 下面为了通用性先将cv2变成任意类型，就是说cv1后接着的可以不是Conv
    if not isinstance(cv2, list):
        cv2 = [cv2]

    for item in cv2:
        if item is None:
            continue

        if isinstance(item, Conv):
            conv = item.conv
        else:
            conv = item

        conv.in_channels = new_feature_num
        # 如上面提到weight的形状是[out, in, k, k], 这里conv2需要对in进行裁剪就是[:, keep_id]
        conv.weight.data = conv.weight.data[:, keep_id]
    
# 处理Conv和C2f的输出和下一层输入
# 下一层可以是C2f和SPPF，或者Conv
def prune(layer1, layer2):
    if isinstance(layer1, C2f):
        cv1 = layer1.cv2 # C2f.cv2
    else:
        cv1 = layer1 # 默认Conv

    if not isinstance(layer2, list):
        cv2 = [layer2]
    else:
        cv2 = layer2

    for i, item in enumerate(cv2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            cv2[i] = item.cv1
        else:
            cv2[i] = item # 默认值Conv

    prune_(cv1, cv2)

def main():
    global threshold
    yolo = YOLO("./stage/before_prune/last.pt")
    model = yolo.model

    ws = []
    bs = []

    # model是nn squential类
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d): 
            # print("layer with bn: ", name)
            w = layer.weight.abs().detach()
            b = layer.bias.abs().detach()
            ws.append(w)
            bs.append(b)

    factor = 0.8
    ws = torch.cat(ws)
    threshold = torch.sort(ws, descending=True)[0][round(len(ws) * factor)]
    print(threshold)
    # 根据阈值，可以对每种层的BN评估剪枝位置并运用在本层组件内的卷积层

####### backbone
# layer with bn:  model.0.bn
# layer with bn:  model.1.bn

# C2f
# layer with bn:  model.2.cv1.bn
# layer with bn:  model.2.cv2.bn
# layer with bn:  model.2.m.0.cv1.bn
# layer with bn:  model.2.m.0.cv2.bn

# layer with bn:  model.3.bn

# C2f
# layer with bn:  model.4.cv1.bn
# layer with bn:  model.4.cv2.bn
# layer with bn:  model.4.m.0.cv1.bn
# layer with bn:  model.4.m.0.cv2.bn

# layer with bn:  model.5.bn

# C2f
# layer with bn:  model.6.cv1.bn
# layer with bn:  model.6.cv2.bn
# layer with bn:  model.6.m.0.cv1.bn
# layer with bn:  model.6.m.0.cv2.bn

# layer with bn:  model.7.bn

# C2f
# layer with bn:  model.8.cv1.bn
# layer with bn:  model.8.cv2.bn
# layer with bn:  model.8.m.0.cv1.bn
# layer with bn:  model.8.m.0.cv2.bn

# SPPF
# layer with bn:  model.9.cv1.bn
# layer with bn:  model.9.cv2.bn

######### head
# C2f
# layer with bn:  model.12.cv1.bn
# layer with bn:  model.12.cv2.bn
# layer with bn:  model.12.m.0.cv1.bn
# layer with bn:  model.12.m.0.cv2.bn

# C2f
# layer with bn:  model.15.cv1.bn
# layer with bn:  model.15.cv2.bn
# layer with bn:  model.15.m.0.cv1.bn
# layer with bn:  model.15.m.0.cv2.bn

# layer with bn:  model.16.bn

# C2f
# layer with bn:  model.18.cv1.bn
# layer with bn:  model.18.cv2.bn
# layer with bn:  model.18.m.0.cv1.bn
# layer with bn:  model.18.m.0.cv2.bn

# layer with bn:  model.19.bn

# C2f
# layer with bn:  model.21.cv1.bn
# layer with bn:  model.21.cv2.bn
# layer with bn:  model.21.m.0.cv1.bn
# layer with bn:  model.21.m.0.cv2.bn

# Detect
# layer with bn:  model.22.cv2.0.0.bn
# layer with bn:  model.22.cv2.0.1.bn
# layer with bn:  model.22.cv2.1.0.bn
# layer with bn:  model.22.cv2.1.1.bn
# layer with bn:  model.22.cv2.2.0.bn
# layer with bn:  model.22.cv2.2.1.bn
# layer with bn:  model.22.cv3.0.0.bn
# layer with bn:  model.22.cv3.0.1.bn
# layer with bn:  model.22.cv3.1.0.bn
# layer with bn:  model.22.cv3.1.1.bn
# layer with bn:  model.22.cv3.2.0.bn
# layer with bn:  model.22.cv3.2.1.bn

# 根据上面print打印得到model包含bn的位置，其中我们可以通过model紧跟的序号快速访问指定位置，或者用isinstance做类型判定直接得到特定的层


    # 针对所有bottleneck, yolo8只有c2f.m有bottleneck
    for name, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            prune_(layer.cv1, layer.cv2)

    # 按指定位置对backbone的输出位置剪枝，并将这些层的后续层输入也做相应裁剪
    # 跳过即4、6、9的输出位置可能是输出位置裁剪会直接影响后续neck的输入参数，修改比较繁琐
    seq = model.model
    print(seq)
    backbone_modify_range = [i for i in range(3, 9)]
    backbone_out_id = [4, 6, 9]
    for i in backbone_modify_range:
        if i in backbone_out_id:
            continue
        prune(seq[i], seq[i + 1])


    # 处理head
    head_out_id = [15, 18, 21] # 出口直接对接detect
    head_out_next = [16, 19, None] # 出口的下一层
    # detect里每个出口分cv2和cv3两个分支，每个分支是Conv+Conv+nn.Conv2d构成
    # 根据上面打印可知有detect与出口对应
    detect : Detect = seq[-1]
    for input, output, cv2, cv3 in zip(head_out_id, head_out_next, detect.cv2, detect.cv3):
        input = seq[input]
        if output is None:
            output = None
        else:
            output = seq[output]

        prune(input, [output, cv2[0], cv3[0]])
        # prune(input, [output, cv2[0], cv3[0]]) 已经做一次剪枝了，这里主要为了下一层能被正确剪枝，以此类推下面的prune(cv2[1], cv2[1])也是如此
        prune(cv2[0], cv2[1]) 
        prune(cv2[1], cv2[2])
        prune(cv3[0], cv3[1])
        prune(cv3[1], cv3[2])

    for name, module in yolo.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"Conv Layer: {name}, Output Channels: {module.out_channels}, Weight Shape: {module.weight.shape}")
        elif isinstance(module, torch.nn.BatchNorm2d):
            print(f"BN Layer: {name}, Num Features: {module.num_features}, Weight Shape: {module.weight.shape}")

    
    for name, p in model.named_parameters():
        p.requires_grad = True
    
    torch.save(yolo.model.state_dict(), "./stage/prune/yolov8n_prune_start_weights.pt")

    # yolo.val(data='data/human_all/data.yaml')
    # yolo.export(format="onnx")
    # yolo.save("./stage/prune/yolov8n_prune_start.pt")
    # torch.save(yolo.ckpt, "./stage/prune/yolov8n_prune_start.pt")
    return

    for name, p in model.named_parameters():
        p.requires_grad = True
    
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} devices")
        device = 'cuda'
        model.cuda()
    else:
        print('CUDA is not available, using CPU')
        device = 'cpu'

    yolo.model = model
    yolo.train(data='data/human_all/data.yaml', epochs=70, device=device)
    yolo.val(data='data/human_all/data.yaml')
    
    yolo.export(format="onnx")

if __name__ == '__main__':
    main()