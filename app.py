
import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TrustLens AI", page_icon="🛡️", layout="wide")

TRAIN_URL = "https://raw.githubusercontent.com/trannguyenthaituan251209/vietnamese_sms_dataset/main/train.csv"
MODEL_PATH = Path("trustlens_model.joblib")

st.markdown("""
<style>
.main {background: #f6f8fc;}
.block-container {padding-top: 2rem; max-width: 1050px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#101f3d,#174c65); color: white;}
.hero h1 {color:white; margin-bottom:0.2rem;}
.small-muted {color:#607087; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang tải dữ liệu và huấn luyện AI lần đầu...")
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.utils import Bunch

    df = pd.read_csv(TRAIN_URL)
    df = df.dropna(subset=["message", "label"])
    df["message"] = df["message"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    X, y = df["message"], df["label"].astype(int)
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents=None)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=90000, sublinear_tf=True))
    ])
    model = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0))
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def risk_signals(message):
    t = message.lower()
    signals = []
    patterns = [
        (r"(yêu cầu|đề nghị|vui lòng|hãy).{0,35}(gửi|cung cấp|đọc|tiết lộ|chia sẻ).{0,25}(otp|mã xác thực|mã xác minh)|(gửi|cung cấp|đọc|tiết lộ|chia sẻ).{0,20}(otp|mã xác thực).{0,30}(cho nhân viên|cho người gọi|qua điện thoại)", "Yêu cầu gửi/chia sẻ mã OTP hoặc mã xác thực"),
        (r"(chuyển khoản|chuyển tiền|nộp tiền|đóng phí).{0,50}(ngay|gấp|khẩn cấp|trong hôm nay)", "Thúc ép chuyển tiền hoặc thanh toán nhanh"),
        (r"(tài khoản|thẻ|sim).{0,50}(bị khóa|bị phong tỏa|bị đình chỉ)", "Đe dọa khóa tài khoản/dịch vụ để tạo áp lực"),
        (r"(công an|tòa án|viện kiểm sát|thuế).{0,70}(chuyển tiền|nộp phạt|xác minh qua link)", "Mạo danh cơ quan chức năng kèm yêu cầu thanh toán/xác minh"),
        (r"(trúng thưởng|nhận thưởng|nhận quà).{0,60}(phí|chuyển khoản|thanh toán)", "Thông báo trúng thưởng nhưng yêu cầu trả phí"),
    ]
    for pattern, explanation in patterns:
        if re.search(pattern, t, flags=re.I | re.S):
            signals.append(explanation)
    urls = re.findall(r"https?://\S+|www\.\S+", message, flags=re.I)
    if urls:
        signals.append("Có đường link; cần kiểm tra tên miền độc lập trước khi mở")
    if re.search(r"\b(bit\.ly|tinyurl|t\.co|is\.gd|cutt\.ly)\b", t):
        signals.append("Có thể sử dụng đường link rút gọn, khó nhận biết đích đến")
    if re.search(r"(không được chậm|ngay lập tức|trong 5 phút|nếu không sẽ|lần cuối)", t):
        signals.append("Ngôn ngữ gây áp lực thời gian hoặc đe dọa")
    return list(dict.fromkeys(signals))

def safe_actions(message, signals, predicted_scam):
    actions = [
        "Không nhấn vào liên kết trong tin nhắn; tự mở ứng dụng hoặc website chính thức bằng địa chỉ đã biết.",
        "Không cung cấp OTP, mật khẩu, mã PIN, CVV hoặc thông tin đăng nhập cho bất kỳ ai.",
        "Nếu có yêu cầu chuyển tiền, hãy gọi cho người/tổ chức qua số điện thoại lấy từ nguồn độc lập để xác minh.",
    ]
    if predicted_scam or signals:
        actions.insert(0, "Tạm dừng giao dịch. Không chuyển tiền cho đến khi xác minh độc lập.")
    return actions

st.markdown("""
<div class="hero">
<h1>🛡️ TrustLens AI</h1>
<p>Nhìn rõ rủi ro trước khi tiền rời khỏi tài khoản.</p>
<p style="opacity:.82">Vietnamese-first AI scam message screening · Explainable signals · Privacy-conscious</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("TrustLens")
    st.caption("Bản thử nghiệm nghiên cứu · AI + Finance")
    st.markdown("**Cách sử dụng**")
    st.write("1. Dán nội dung tin nhắn hoặc yêu cầu thanh toán.")
    st.write("2. Nhấn Phân tích.")
    st.write("3. Đọc mức cảnh báo, dấu hiệu và bước xác minh.")
    st.divider()
    st.caption("Tin nhắn được xử lý trong phiên chạy hiện tại và không được lưu bởi ứng dụng này.")
    st.caption("Không nhập OTP, mật khẩu, số thẻ hoặc dữ liệu cá nhân thật.")

st.subheader("Kiểm tra tin nhắn")
sample = st.selectbox("Thử nhanh một ví dụ", [
    "Tự nhập nội dung",
    "Ví dụ đáng ngờ",
    "Ví dụ thông báo hợp lệ"
])
defaults = {
    "Tự nhập nội dung": "",
    "Ví dụ đáng ngờ": "CONG AN: Tai khoan cua ban lien quan vu an. De xac minh, hay chuyen 15.000.000 dong vao tai khoan tam giu trong 30 phut va gui ma OTP qua duong link http://xacminh-example.com",
    "Ví dụ thông báo hợp lệ": "Ma OTP dang nhap ung dung cua ban la [NUMBER]. Tuyet doi khong chia se ma nay voi bat ky ai, ke ca nhan vien ngan hang."
}
message = st.text_area("Nội dung cần kiểm tra", value=defaults[sample], height=170,
                       placeholder="Dán tin nhắn SMS, chat hoặc yêu cầu thanh toán tại đây...")
col1, col2 = st.columns([1, 3])
with col1:
    analyse = st.button("🔎 Phân tích rủi ro", type="primary", use_container_width=True)
with col2:
    st.caption("Không cần nhập tên, số điện thoại, tài khoản ngân hàng hoặc thông tin định danh.")

if analyse:
    if not message.strip():
        st.warning("Vui lòng nhập nội dung trước khi phân tích.")
    else:
        try:
            model = load_model()
            probability = float(model.predict_proba([message])[0][list(model.classes_).index(1)])
            signals = risk_signals(message)
            # A conservative review threshold: a warning is not proof of fraud.
            if probability >= 0.70 or (probability >= 0.45 and len(signals) >= 1):
                status, color, label = "NGUY CƠ CAO — CẦN XÁC MINH", "error", "Không nên thực hiện giao dịch"
                predicted_scam = True
            elif probability >= 0.35 or signals:
                status, color, label = "CẦN THẬN TRỌNG", "warning", "Tạm dừng và xác minh qua kênh độc lập"
                predicted_scam = False
            else:
                status, color, label = "CHƯA PHÁT HIỆN DẤU HIỆU RÕ RÀNG", "success", "Không đồng nghĩa với an toàn tuyệt đối"
                predicted_scam = False

            st.divider()
            st.subheader("Kết quả phân tích")
            a, b = st.columns([1, 2])
            with a:
                st.metric("Điểm mô hình", f"{probability*100:.1f}/100")
                st.progress(min(max(probability, 0.0), 1.0))
                st.caption("Điểm mô hình không phải xác suất đã hiệu chuẩn.")
            with b:
                if color == "error":
                    st.error(status)
                elif color == "warning":
                    st.warning(status)
                else:
                    st.success(status)
                st.write(label)

            st.markdown("#### Dấu hiệu cần chú ý")
            if signals:
                for s in signals:
                    st.write("• " + s)
            else:
                st.write("Chưa phát hiện dấu hiệu theo bộ quy tắc minh bạch. Mô hình vẫn có thể bỏ sót thủ đoạn mới.")

            st.markdown("#### Việc nên làm ngay")
            for action in safe_actions(message, signals, predicted_scam):
                st.write("☐ " + action)

            st.info("Giới hạn: mô hình học từ tập SMS công khai có nhãn. Nó có thể nhầm quảng cáo hợp lệ với spam, bỏ sót lừa đảo mới, hoặc đánh giá sai nội dung ngoài miền SMS. Không dùng kết quả này làm căn cứ duy nhất để chuyển tiền, khóa thẻ hay kết luận một cá nhân phạm tội.")
        except Exception as e:
            st.error("Không thể tải mô hình/dữ liệu. Hãy kiểm tra kết nối Internet rồi tải lại ứng dụng.")
            st.exception(e)

st.divider()
st.markdown("**Nguyên tắc TrustLens:** AI đưa ra tín hiệu để người dùng kiểm tra kỹ hơn — không thay thế ngân hàng, cơ quan chức năng hoặc phán đoán của con người.")
st.caption("Prototype · Không phải tư vấn tài chính. Nếu đã chuyển tiền hoặc tiết lộ thông tin, liên hệ ngân hàng qua kênh chính thức ngay lập tức.")
