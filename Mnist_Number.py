import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

st.markdown("""
<style>
.hero{
    padding:25px;
    border-radius:18px;
    background:linear-gradient(90deg,#4F46E5,#2563EB);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.stButton>button{
    background:#2563EB;
    color:white;
    height:45px;
    width:100%;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
<h1>🔢 MNIST Digit Classification</h1>
<p>Upload an image (0-9) and predict the digit using CNN.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    return mnist.load_data()

(x_train, y_train), (x_test, y_test) = load_dataset()

@st.cache_resource
def train_model():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    model = Sequential([

        Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
        MaxPooling2D(2,2),

        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(64,activation="relu"),
        Dense(32,activation="relu"),

        Dense(10,activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    return model, x_test, y_test

with st.spinner("Training CNN Model... Please wait..."):
    model, x_test, y_test = train_model()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

st.success("✅ Model Ready")

with st.sidebar:
    st.title("📌 Model Information")
    st.success("Dataset : MNIST")
    st.info("Images : 70,000")
    st.info("Classes : 10")
    st.info("Image Size : 28 × 28")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

file = st.file_uploader(
    "📂 Upload Digit Image",
    type=["png","jpg","jpeg"]
)

if file is not None:

    image = tf.keras.preprocessing.image.load_img(
        file,
        target_size=(28,28),
        color_mode="grayscale"
    )

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0

    # Invert image (recommended for white background images)
    image_array = 1.0 - image_array

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image_array[0].reshape(28,28),
            width=220,
            clamp=True
        )

    with col2:
        st.subheader("Prediction")
        st.success(f"Predicted Digit : {predicted_digit}")
        st.progress(float(confidence))
        st.write(f"Confidence : **{confidence*100:.2f}%**")

else:
    st.info("Please upload a digit image.")

st.markdown("---")
st.caption("Developed using TensorFlow + Streamlit")
