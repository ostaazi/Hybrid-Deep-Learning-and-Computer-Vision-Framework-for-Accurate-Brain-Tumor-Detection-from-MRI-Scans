from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# ------------------------ إعدادات المسارات ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------------ تحميل النموذج ------------------------
# هذا هو نفس النموذج الأصلي المدرب مسبقًا
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
model = load_model(MODEL_PATH)

# ترتيب الأصناف كما هو في الكود الأصلي
# (مهم جدًا حتى تكون النتائج صحيحة)
CLASS_KEYS = ["pituitary", "glioma", "notumor", "meningioma"]

# أسماء للعرض في الواجهة
DISPLAY_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor",
}


# ------------------------ دالة التنبؤ ------------------------
def predict_probabilities(image_path: str):
    """
    ترجع مصفوفة احتمالات النموذج مرتبة مثل CLASS_KEYS
    مع قيم بين 0 و 1
    """
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]  # شكلها (4,)
    return preds  # numpy array


# ------------------------ المسارات ------------------------
@app.route("/", methods=["GET"])
def index():
    # الواجهة المتقدمة (BrainVision AI)
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify(
                {"success": False, "error": "No file part in the request."}
            ), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify(
                {"success": False, "error": "No file selected."}
            ), 400

        # حفظ الصورة في مجلد uploads (مثل الكود الأصلي)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        # الحصول على الاحتمالات من النموذج
        probs = predict_probabilities(save_path)  # (4,)
        probs = probs.astype(float)

        # أعلى صنف
        top_idx = int(np.argmax(probs))
        top_key = CLASS_KEYS[top_idx]
        top_label = DISPLAY_LABELS[top_key]
        top_confidence = float(probs[top_idx]) * 100.0  # نسبة مئوية

        # تحويل كل الاحتمالات إلى قاموس للواجهة
        probabilities = {}
        for key in CLASS_KEYS:
            display_name = DISPLAY_LABELS[key]
            idx = CLASS_KEYS.index(key)
            probabilities[display_name] = float(probs[idx]) * 100.0

        return jsonify(
            {
                "success": True,
                "top_class": top_label,
                "confidence": round(top_confidence, 2),
                "probabilities": {
                    name: round(val, 2) for name, val in probabilities.items()
                },
            }
        )

    except Exception as e:
        # إرسال رسالة الخطأ إلى الواجهة
        return jsonify(
            {
                "success": False,
                "error": f"Internal error during prediction: {str(e)}",
            }
        ), 500


if __name__ == "__main__":
    # 8000 حتى تعمل بسهولة على GitHub Codespaces / VS Code
    app.run(host="0.0.0.0", port=8000, debug=True)
