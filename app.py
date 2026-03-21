import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ====================== YOLOv8配置（添加缓存） ======================
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO('yolov8n.pt')

YOLO_MODEL = load_yolo_model()
CAT_CLASS_ID = 15
YOLO_DETECT_THRESH = 0.5

# ====================== 全局配置（修改模型路径+手机适配） ======================
CAT_CLASS_NAMES = [
    '阿比西尼亚猫', '孟加拉猫', '伯曼猫', '孟买猫', '英国短毛猫',
    '埃及猫', '缅因猫', '波斯猫', '布偶猫', '俄罗斯蓝猫',
    '暹罗猫', '无毛猫'
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 改为相对路径（需上传trained_efficientnet.pth到GitHub）
CAT_BREED_MODEL_PATH = "trained_efficientnet.pth"

# ====================== 1. 加载模型 ======================
@st.cache_resource(show_spinner="加载品种分类模型...")
def load_cat_breed_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 12)
    )

    state_dict = torch.load(CAT_BREED_MODEL_PATH, map_location=DEVICE)
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

# ====================== 2. YOLO检测（返回原图/裁剪图） ======================
def yolo_detect_cat(image):
    results = YOLO_MODEL(image, conf=YOLO_DETECT_THRESH)
    cat_boxes = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls) == CAT_CLASS_ID:
                    cat_boxes.append(box.xyxy.cpu().numpy()[0].astype(int))
    is_cat = len(cat_boxes) > 0
    use_image = image  # 未检测到猫时用原图
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

# ====================== 3. 预处理 ======================
def preprocess_image(image):
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

# ====================== 4. 预测函数 ======================
def predict_cat_breed(model, image):
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

# ====================== 5. 可视化（适配手机） ======================
def plot_prediction(probabilities, top5_classes, top5_probs):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))  # 缩小图表适配手机
    colors = sns.color_palette('viridis', 5)
    ax1.barh(top5_classes[::-1], top5_probs[::-1], color=colors)
    ax1.set_xlabel('置信度', fontsize=12)
    ax1.set_title('前5名品种预测结果', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    for i, (cls, prob) in enumerate(zip(top5_classes[::-1], top5_probs[::-1])):
        ax1.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=10)
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
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('置信度', fontsize=10)
    plt.tight_layout()
    return fig

# ====================== 6. 绘制检测框 ======================
def draw_detection_box(image, cat_boxes):
    draw = ImageDraw.Draw(image)
    for box in cat_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1 - 10), '猫', fill='red', fontsize=16)
    return image

# ====================== 7. 主界面（手机适配） ======================
def main():
    st.set_page_config(
        page_title="猫品种智能分类系统（强制识别版）",
        page_icon="🐱",
        layout="centered",  # 手机端居中
        initial_sidebar_state="collapsed"  # 隐藏侧边栏
    )

    st.title("🐱 猫品种智能分类系统（强制识别版）")
    st.markdown("---")

    # 侧边栏（手机端可折叠）
    with st.sidebar:
        st.header("功能说明")
        st.success("""
        🚀 正常识别：检测到猫 → 自动识别品种
        ⚠️ 强制识别：未检测到猫时，可手动选择用整张图识别（结果仅供参考）
        🎯 清空结果：一键重置，准备下一次识别
        """)
        st.markdown("---")
        st.subheader("模型参数")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("YOLO检测阈值", "50%")
        with col_s2:
            st.metric("分类模型", "EfficientNetB0")

    # 主界面（手机端适配）
    col1, col2 = st.columns([1, 1.2])  # 调整列宽适配手机
    breed_model = load_cat_breed_model()  # 提前加载模型

    with col1:
        st.subheader("上传图片")
        uploaded_file = st.file_uploader(
            "支持格式：jpg/png/webp",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=False
        )

        # 清空结果按钮（用户主动重置）
        if st.button("🗑️ 清空结果", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ 已清空所有识别结果！")

        if uploaded_file is None:
            # 无图片时清空状态
            for key in list(st.session_state.keys()):
                del st.session_state[key]
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)

            # 第一步：YOLO检测
            with st.spinner("🔍 正在检测猫的位置..."):
                is_cat, cat_boxes, use_image = yolo_detect_cat(image)

                if is_cat:
                    # 正常识别：检测到猫
                    st.success(f"✅ 检测到 {len(cat_boxes)} 只猫！")
                    image_with_box = image.copy()
                    image_with_box = draw_detection_box(image_with_box, cat_boxes)
                    st.image(image_with_box, caption="猫的检测区域", use_container_width=True)
                    st.image(use_image, caption="裁剪后的猫区域", use_container_width=True)

                    # 自动识别品种
                    with st.spinner("🧠 正在识别猫品种..."):
                        results = predict_cat_breed(breed_model, use_image)
                        st.success(f"🐱 识别结果：{results['best_class']}")
                        st.metric("置信度", f"{results['best_prob']:.2%}")
                        st.session_state = {
                            'is_cat': True,
                            'breed_results': results,
                            'cat_boxes': cat_boxes,
                            'force_mode': False
                        }
                else:
                    # 未检测到猫：显示强制识别按钮
                    st.warning("⚠️ 未检测到猫！")
                    # 强制识别按钮
                    if st.button("🔨 强制识别（结果仅供参考）", type="primary"):
                        with st.spinner("🧠 正在强制识别品种..."):
                            results = predict_cat_breed(breed_model, use_image)  # 用整张图识别
                            st.warning(f"⚠️ 强制识别结果：{results['best_class']}（未检测到猫，结果仅供参考）")
                            st.metric("置信度", f"{results['best_prob']:.2%}", help="未检测到猫，结果可能不准确")
                            st.session_state = {
                                'is_cat': False,
                                'breed_results': results,
                                'cat_boxes': [],
                                'force_mode': True
                            }

    with col2:
        st.subheader("识别结果可视化")
        if 'breed_results' in st.session_state:
            results = st.session_state['breed_results']
            fig = plot_prediction(results['all_probs'], results['top5_classes'], results['top5_probs'])
            st.pyplot(fig, use_container_width=True)  # 适配手机宽度

            st.markdown("### 📊 详细预测结果")
            st.table({
                "品种": results['top5_classes'],
                "置信度": [f"{p:.2%}" for p in results['top5_probs']]
            })

            with st.expander("🔧 技术详情"):
                force_note = "（强制识别模式）" if st.session_state.get('force_mode', False) else ""
                st.write(f"""
                - 检测模型：YOLOv8n（轻量级）
                - 分类模型：EfficientNetB0（迁移学习）{force_note}
                - 运行设备：{DEVICE}
                - 检测到猫数量：{len(st.session_state.get('cat_boxes', []))}
                - 模型路径：{CAT_BREED_MODEL_PATH}
                """)
        else:
            if uploaded_file is None:
                st.info("📌 上传图片后可选择：\n1. 正常识别（检测到猫自动执行）\n2. 强制识别（未检测到猫手动触发）")
            else:
                st.info("⚠️ 未检测到猫，点击「强制识别」可尝试用整张图识别")

    st.markdown("---")
    st.markdown("© 2026 猫品种分类系统 | 支持正常识别/强制识别/清空结果")

if __name__ == "__main__":
    main()