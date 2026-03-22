import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import requests
from io import BytesIO

warnings.filterwarnings('ignore')

# ====================== 全局配置 ======================
CAT_CLASS_NAMES = [
    '阿比西尼亚猫', '孟加拉猫', '伯曼猫', '孟买猫', '英国短毛猫',
    '埃及猫', '缅因猫', '波斯猫', '布偶猫', '俄罗斯蓝猫',
    '暹罗猫', '无毛猫'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型文件路径（Streamlit Cloud使用当前目录）
CAT_BREED_MODEL_PATH = "trained_efficientnet.pth"
YOLO_MODEL_PATH = "yolov8n.pt"
CAT_CLASS_ID = 15
YOLO_DETECT_THRESH = 0.5

# ====================== 下载模型文件（如果不存在） ======================
def download_model(url, path):
    """从URL下载模型文件"""
    if not os.path.exists(path):
        with st.spinner(f"正在下载模型文件 {path}..."):
            response = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"✓ {path} 下载完成")

@st.cache_resource
def load_yolo_model():
    """加载YOLO模型"""
    from ultralytics import YOLO
    
    # 如果模型不存在，从官方源下载
    if not os.path.exists(YOLO_MODEL_PATH):
        download_model(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            YOLO_MODEL_PATH
        )
    
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource(show_spinner="加载品种分类模型...")
def load_cat_breed_model():
    """加载猫品种分类模型"""
    # 检查模型文件是否存在
    if not os.path.exists(CAT_BREED_MODEL_PATH):
        st.error(f"❌ 模型文件 {CAT_BREED_MODEL_PATH} 不存在！")
        st.info("请确保已将模型文件上传到仓库根目录")
        return None
    
    # 创建模型结构
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 12)
    )
    
    # 加载权重
    try:
        state_dict = torch.load(CAT_BREED_MODEL_PATH, map_location=DEVICE)
        
        # 处理模型结构映射
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "classifier.3.weight":
                new_state_dict["classifier.1.3.weight"] = v
            elif k == "classifier.3.bias":
                new_state_dict["classifier.1.3.bias"] = v
            elif k == "classifier.1.weight":
                new_state_dict["classifier.1.0.weight"] = v
            elif k == "classifier.1.bias":
                new_state_dict["classifier.1.0.bias"] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# ====================== YOLO检测 ======================
def yolo_detect_cat(image):
    """检测图片中的猫"""
    model = load_yolo_model()
    results = model(image, conf=YOLO_DETECT_THRESH)
    cat_boxes = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls) == CAT_CLASS_ID:
                    cat_boxes.append(box.xyxy.cpu().numpy()[0].astype(int))
    
    is_cat = len(cat_boxes) > 0
    use_image = image
    
    if is_cat:
        x1, y1, x2, y2 = cat_boxes[0]
        w, h = image.size
        expand_w = int((x2 - x1) * 0.1)
        expand_h = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)
        use_image = image.crop((x1, y1, x2, y2))
    
    return is_cat, cat_boxes, use_image

# ====================== 预处理 ======================
def preprocess_image(image):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return img_tensor

# ====================== 预测 ======================
def predict_cat_breed(model, image):
    """预测猫品种"""
    if model is None:
        return None
    
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top5_idx = np.argsort(probabilities)[::-1][:5]
    top5_probs = probabilities[top5_idx]
    top5_classes = [CAT_CLASS_NAMES[i] for i in top5_idx]
    best_idx = np.argmax(probabilities)
    best_class = CAT_CLASS_NAMES[best_idx]
    best_prob = probabilities[best_idx]
    
    return {
        'best_class': best_class,
        'best_prob': best_prob,
        'top5_classes': top5_classes,
        'top5_probs': top5_probs,
        'all_probs': probabilities
    }

# ====================== 可视化 ======================
def plot_prediction(probabilities, top5_classes, top5_probs):
    """绘制预测结果图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = sns.color_palette('viridis', 5)
    
    # 水平条形图
    ax1.barh(top5_classes[::-1], top5_probs[::-1], color=colors)
    ax1.set_xlabel('置信度', fontsize=12)
    ax1.set_title('Top 5 预测结果', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    
    for i, (cls, prob) in enumerate(zip(top5_classes[::-1], top5_probs[::-1])):
        ax1.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=10)
    
    # 热力图
    im = ax2.imshow(probabilities.reshape(3, 4), cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            if idx < 12:
                ax2.text(j, i, f'{CAT_CLASS_NAMES[idx]}\n{probabilities[idx]:.1%}',
                         ha='center', va='center', fontsize=8, color='black')
    
    ax2.set_title('所有品种置信度分布', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    plt.tight_layout()
    
    return fig

# ====================== 绘制检测框 ======================
def draw_detection_box(image, cat_boxes):
    """绘制检测框"""
    draw = ImageDraw.Draw(image)
    for box in cat_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1 - 10), '猫', fill='red')
    return image

# ====================== 主界面 ======================
def main():
    st.set_page_config(
        page_title="猫品种智能分类系统",
        page_icon="🐱",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # 显示设备信息
    st.sidebar.info(f"运行设备: {DEVICE}")
    
    st.title("🐱 猫品种智能分类系统")
    st.markdown("---")
    
    # 显示模型状态
    model = load_cat_breed_model()
    if model is None:
        st.error("❌ 模型加载失败，请检查配置文件")
        return
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传猫的图片",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="支持JPG、PNG、WEBP格式"
    )
    
    if uploaded_file is not None:
        # 显示图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        # 检测按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 识别猫品种", type="primary", use_container_width=True):
                with st.spinner("正在分析图片..."):
                    # YOLO检测
                    is_cat, cat_boxes, use_image = yolo_detect_cat(image)
                    
                    if is_cat:
                        st.success(f"✅ 检测到 {len(cat_boxes)} 只猫！")
                        
                        # 显示检测框
                        image_with_box = image.copy()
                        image_with_box = draw_detection_box(image_with_box, cat_boxes)
                        st.image(image_with_box, caption="猫的检测区域", use_container_width=True)
                        st.image(use_image, caption="裁剪后的猫区域", use_container_width=True)
                        
                        # 品种识别
                        results = predict_cat_breed(model, use_image)
                        if results:
                            st.success(f"🐱 识别结果：**{results['best_class']}**")
                            st.metric("置信度", f"{results['best_prob']:.2%}")
                            
                            # 显示图表
                            fig = plot_prediction(
                                results['all_probs'],
                                results['top5_classes'],
                                results['top5_probs']
                            )
                            st.pyplot(fig, use_container_width=True)
                            
                            # 详细结果表格
                            st.markdown("### 📊 详细预测结果")
                            st.table({
                                "品种": results['top5_classes'],
                                "置信度": [f"{p:.2%}" for p in results['top5_probs']]
                            })
                    else:
                        st.warning("⚠️ 未检测到猫！")
                        if st.button("🔄 强制识别（使用整张图片）"):
                            results = predict_cat_breed(model, image)
                            if results:
                                st.info(f"强制识别结果：**{results['best_class']}**")
                                st.metric("置信度", f"{results['best_prob']:.2%}")
                                st.warning("注意：未检测到猫，结果仅供参考")
        
        with col2:
            if st.button("🗑️ 清空结果", use_container_width=True):
                st.rerun()
    
    # 页脚
    st.markdown("---")
    st.markdown("© 2026 猫品种分类系统 | 基于YOLOv8 + EfficientNet")

if __name__ == "__main__":
    main()
