import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ====================== 全局配置（移除YOLO，保留核心配置） ======================
CAT_CLASS_NAMES = [
    '阿比西尼亚猫', '孟加拉猫', '伯曼猫', '孟买猫', '英国短毛猫',
    '埃及猫', '缅因猫', '波斯猫', '布偶猫', '俄罗斯蓝猫',
    '暹罗猫', '无毛猫'
]
DEVICE = torch.device("cpu")  # 强制CPU，适配Streamlit Cloud
CAT_BREED_MODEL_PATH = "trained_efficientnet.pth"  # 保持相对路径

# ====================== 1. 加载品种分类模型（保留核心） ======================
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

    # 加载模型权重（适配CPU）
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

# ====================== 2. 预处理（保留核心） ======================
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

# ====================== 3. 预测函数（保留核心） ======================
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

# ====================== 4. 可视化（适配手机，保留核心） ======================
def plot_prediction(probabilities, top5_classes, top5_probs):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))  # 适配手机
    colors = sns.color_palette('viridis', 5)
    
    # 前5名柱状图
    ax1.barh(top5_classes[::-1], top5_probs[::-1], color=colors)
    ax1.set_xlabel('置信度', fontsize=12)
    ax1.set_title('前5名品种预测结果', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    for i, (cls, prob) in enumerate(zip(top5_classes[::-1], top5_probs[::-1])):
        ax1.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=10)
    
    # 所有品种热力图
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

# ====================== 5. 主界面（移除YOLO，保留核心交互） ======================
def main():
    st.set_page_config(
        page_title="猫品种智能分类系统",
        page_icon="🐱",
        layout="centered",  # 手机端居中
        initial_sidebar_state="collapsed"  # 隐藏侧边栏
    )

    st.title("🐱 猫品种智能分类系统")
    st.markdown("---")

    # 侧边栏（简化说明）
    with st.sidebar:
        st.header("功能说明")
        st.success("""
        🚀 上传猫咪图片即可识别品种
        🗑️ 清空结果：一键重置，准备下一次识别
        🎯 支持查看前5名预测结果及置信度分布
        """)
        st.markdown("---")
        st.subheader("模型参数")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("分类模型", "EfficientNetB0")
        with col_s2:
            st.metric("运行设备", "CPU")

    # 主界面（适配手机）
    col1, col2 = st.columns([1, 1.2])
    breed_model = load_cat_breed_model()  # 提前加载模型

    with col1:
        st.subheader("上传图片")
        uploaded_file = st.file_uploader(
            "支持格式：jpg/png/webp",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=False
        )

        # 清空结果按钮
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

            # 直接识别品种（移除YOLO检测步骤）
            with st.spinner("🧠 正在识别猫品种..."):
                results = predict_cat_breed(breed_model, image)
                st.success(f"🐱 识别结果：{results['best_class']}")
                st.metric("置信度", f"{results['best_prob']:.2%}")
                st.session_state = {
                    'breed_results': results,
                    'uploaded_image': image
                }

    with col2:
        st.subheader("识别结果可视化")
        if 'breed_results' in st.session_state:
            results = st.session_state['breed_results']
            fig = plot_prediction(results['all_probs'], results['top5_classes'], results['top5_probs'])
            st.pyplot(fig, use_container_width=True)

            # 详细结果表格
            st.markdown("### 📊 详细预测结果")
            st.table({
                "品种": results['top5_classes'],
                "置信度": [f"{p:.2%}" for p in results['top5_probs']]
            })

            # 技术详情
            with st.expander("🔧 技术详情"):
                st.write(f"""
                - 分类模型：EfficientNetB0（迁移学习）
                - 运行设备：{DEVICE}
                - 模型路径：{CAT_BREED_MODEL_PATH}
                - 支持品种数：12种常见猫品种
                """)
        else:
            st.info("📌 上传猫咪图片即可自动识别品种，并展示详细的置信度分布")

    st.markdown("---")
    st.markdown("© 2026 猫品种分类系统 | 适配Streamlit Cloud部署")

if __name__ == "__main__":
    main()
