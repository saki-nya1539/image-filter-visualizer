import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="画像処理フィルタ可視化ツール",
    layout="wide"
)

st.title("画像処理フィルタ可視化ツール")
st.write("画像をアップロードして、画像処理フィルタの効果を比較できます。")

uploaded_file = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

def pil_to_cv2(pil_image):
    image = np.array(pil_image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def cv2_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_histogram(image):
    fig, ax = plt.subplots()

    if len(image.shape) == 2:
        ax.hist(image.ravel(), bins=256, range=(0, 256), color="gray")
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = ("red", "green", "blue")

        for i, color in enumerate(colors):
            ax.hist(
                rgb[:, :, i].ravel(),
                bins=256,
                range=(0, 256),
                color=color,
                alpha=0.5
            )

    ax.set_title("Histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def apply_filter(image, filter_name, threshold_value, canny_low, canny_high):
    if filter_name == "グレースケール":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif filter_name == "二値化":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )
        return binary

    elif filter_name == "ガウシアンぼかし":
        return cv2.GaussianBlur(image, (9, 9), 0)

    elif filter_name == "メディアンフィルタ":
        return cv2.medianBlur(image, 5)

    elif filter_name == "Sobelエッジ検出":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        return sobel

    elif filter_name == "Laplacianエッジ検出":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        return laplacian

    elif filter_name == "Cannyエッジ検出":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, canny_low, canny_high)

    elif filter_name == "シャープ化":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(image, -1, kernel)

    else:
        return image

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original = pil_to_cv2(pil_image)

    st.sidebar.header("設定")

    filter_name = st.sidebar.selectbox(
        "フィルタを選択",
        [
            "グレースケール",
            "二値化",
            "ガウシアンぼかし",
            "メディアンフィルタ",
            "Sobelエッジ検出",
            "Laplacianエッジ検出",
            "Cannyエッジ検出",
            "シャープ化"
        ]
    )

    threshold_value = st.sidebar.slider(
        "二値化の閾値",
        0,
        255,
        127
    )

    canny_low = st.sidebar.slider(
        "Canny下限閾値",
        0,
        255,
        100
    )

    canny_high = st.sidebar.slider(
        "Canny上限閾値",
        0,
        255,
        200
    )

    start_time = time.time()
    processed = apply_filter(
        original,
        filter_name,
        threshold_value,
        canny_low,
        canny_high
    )
    elapsed_time = time.time() - start_time

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("元画像")
        st.image(cv2_to_rgb(original), use_container_width=True)

    with col2:
        st.subheader(f"処理後画像：{filter_name}")

        if len(processed.shape) == 2:
            st.image(processed, clamp=True, use_container_width=True)
        else:
            st.image(cv2_to_rgb(processed), use_container_width=True)

    st.subheader("処理情報")
    st.write(f"処理時間：{elapsed_time:.5f} 秒")

    st.subheader("ヒストグラム")
    show_histogram(processed)

else:
    st.info("まず画像をアップロードしてください。")