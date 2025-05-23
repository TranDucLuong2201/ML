# =============================================================================
# GIAO DIỆN DỰ ĐOÁN LOẠI RƯỢU CHO JUPYTER NOTEBOOK
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import warnings
warnings.filterwarnings('ignore')

class WineModelPredictor:
    """
    Lớp dự đoán loại rượu tối ưu cho Jupyter Notebook với widgets
    """
    def __init__(self):
        self.lr_model = None
        self.nb_model = None
        self.feature_names = [
            'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue',
            'OD280/OD315 of diluted wines', 'Proline'
        ]
        self.feature_ranges = {
            'Alcohol': (11.0, 15.0),
            'Malic acid': (0.7, 6.0),
            'Ash': (1.4, 3.3),
            'Alcalinity of ash': (10.6, 30.0),
            'Magnesium': (70, 162),
            'Total phenols': (0.98, 4.0),
            'Flavanoids': (0.34, 5.08),
            'Nonflavanoid phenols': (0.13, 0.66),
            'Proanthocyanins': (0.41, 3.58),
            'Color intensity': (1.28, 13.0),
            'Hue': (0.48, 1.71),
            'OD280/OD315 of diluted wines': (1.27, 4.0),
            'Proline': (278, 1680)
        }
        self.model_info = None
        self.input_widgets = {}
        self.output_widget = None
        self.wine_classes = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

    def load_models(self, lr_path='best_logistic_regression_model.pkl',
                   nb_path='naive_bayes_model.pkl',
                   results_path='model_comparison_results.pkl'):
        """Load các mô hình đã được huấn luyện"""
        try:
            print("🔄 Đang tải các mô hình...")
            self.lr_model = joblib.load(lr_path)
            self.nb_model = joblib.load(nb_path)
            self.model_info = joblib.load(results_path)
            print("✅ Đã tải thành công các mô hình!")
            self.display_model_info()
            return True
        except FileNotFoundError as e:
            print(f"❌ Không tìm thấy file: {e}")
            print("💡 Chạy code huấn luyện trước hoặc sử dụng phương thức demo!")
            return False
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def display_model_info(self):
        """Hiển thị thông tin mô hình"""
        if not self.model_info:
            return

        lr_info = self.model_info['logistic_regression']
        nb_info = self.model_info['naive_bayes']

        html_content = f"""
        <div style="background: linear-gradient(135deg, #8B0000, #DC143C); padding: 20px; border-radius: 15px; margin: 15px 0; color: white;">
            <h3>🍷 THÔNG TIN MÔ HÌNH DỰ ĐOÁN LOẠI RƯỢU</h3>
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div style="flex: 1; background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
                    <h4>🔵 Logistic Regression</h4>
                    <p><strong>F1-Score:</strong> {lr_info['metrics']['f1_weighted']:.4f}</p>
                    <p><strong>Accuracy:</strong> {lr_info['metrics']['accuracy']:.4f}</p>
                    <p><strong>Tham số:</strong> {lr_info['best_params']}</p>
                </div>
                <div style="flex: 1; background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
                    <h4>🟢 Naive Bayes</h4>
                    <p><strong>F1-Score:</strong> {nb_info['metrics']['f1_weighted']:.4f}</p>
                    <p><strong>Accuracy:</strong> {nb_info['metrics']['accuracy']:.4f}</p>
                    <p><strong>Tham số:</strong> {nb_info['best_params']}</p>
                </div>
            </div>
        </div>
        """
        display(HTML(html_content))

    def setup_demo_models(self):
        """Thiết lập mô hình demo với wine dataset"""
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        print("🍷 Tạo mô hình demo với Wine Dataset...")

        # Load wine dataset
        wine_data = load_wine()
        X, y = wine_data.data, wine_data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tạo và huấn luyện mô hình
        self.lr_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        self.nb_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])

        self.lr_model.fit(X_train, y_train)
        self.nb_model.fit(X_train, y_train)

        # Cập nhật tên classes
        self.wine_classes = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

        print("✅ Đã tạo mô hình demo với Wine Dataset thành công!")
        return True

    def create_input_widgets(self):
        """Tạo widgets để nhập 13 đặc trưng của rượu"""
        self.input_widgets = {}

        for feature in self.feature_names:
            min_val, max_val = self.feature_ranges[feature]
            default_val = (min_val + max_val) / 2

            self.input_widgets[feature] = widgets.FloatText(
                value=round(default_val, 2),
                description=f'{feature}:',
                style={'description_width': '200px'},
                layout=widgets.Layout(width='350px'),
                step=0.01
            )

        # Nút dự đoán
        self.predict_button = widgets.Button(
            description='🍷 Dự đoán loại rượu',
            button_style='danger',
            layout=widgets.Layout(width='250px', height='50px'),
            style={'font_weight': 'bold'}
        )

        self.predict_button.on_click(self.on_predict_click)

        # Nút reset
        self.reset_button = widgets.Button(
            description='🔄 Reset',
            button_style='warning',
            layout=widgets.Layout(width='100px', height='50px')
        )

        self.reset_button.on_click(self.on_reset_click)

        # Widget hiển thị kết quả
        self.output_widget = widgets.Output()

        return self.input_widgets

    def display_input_interface(self):
        """Hiển thị giao diện nhập liệu cho 13 đặc trưng"""
        if not self.input_widgets:
            self.create_input_widgets()

        # Header
        header = widgets.HTML("""
        <div style="background: linear-gradient(135deg, #722F37, #C73E1D);
                    color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <h2>🍷 DỰ ĐOÁN LOẠI RƯỢU VỚI MACHINE LEARNING</h2>
            <p style="font-size: 16px; margin-top: 10px;">Nhập 13 đặc trưng hóa học để dự đoán loại rượu</p>
            <p style="font-size: 14px; opacity: 0.9;">Sử dụng Logistic Regression và Naive Bayes</p>
        </div>
        """)

        # Hướng dẫn và thông tin về features
        feature_info = widgets.HTML("""
        <div style="background-color: #FFF8DC; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #DAA520;">
            <h4>📖 Thông tin về các đặc trưng:</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px;">
                <div><strong>Alcohol:</strong> Độ cồn (%)</div>
                <div><strong>Color intensity:</strong> Cường độ màu sắc</div>
                <div><strong>Malic acid:</strong> Axit malic (g/L)</div>
                <div><strong>Hue:</strong> Sắc độ</div>
                <div><strong>Ash:</strong> Tro (g/L)</div>
                <div><strong>OD280/OD315:</strong> Tỷ lệ quang học</div>
                <div><strong>Alcalinity of ash:</strong> Độ kiềm của tro</div>
                <div><strong>Proline:</strong> Proline (mg/L)</div>
                <div><strong>Magnesium:</strong> Magie (mg/L)</div>
                <div><strong>Total phenols:</strong> Tổng phenol</div>
                <div><strong>Flavanoids:</strong> Flavanoid</div>
                <div><strong>Nonflavanoid phenols:</strong> Phenol không phải flavanoid</div>
                <div><strong>Proanthocyanins:</strong> Proanthocyanin</div>
            </div>
        </div>
        """)

        # Tạo layout cho 13 inputs (3 cột)
        input_rows = []
        for i in range(0, len(self.feature_names), 3):
            features_in_row = self.feature_names[i:i+3]
            widgets_in_row = [self.input_widgets[feature] for feature in features_in_row]

            # Thêm placeholder nếu không đủ 3 widgets
            while len(widgets_in_row) < 3:
                widgets_in_row.append(widgets.HTML(""))

            row = widgets.HBox(widgets_in_row, layout=widgets.Layout(margin='5px 0'))
            input_rows.append(row)

        # Buttons
        button_row = widgets.HBox([
            self.predict_button,
            self.reset_button
        ], layout=widgets.Layout(justify_content='center', margin='20px 0'))

        # Container chính
        main_container = widgets.VBox([
            header,
            feature_info,
            widgets.HTML("<h3 style='color: #722F37;'>📝 Nhập các đặc trưng hóa học:</h3>"),
            *input_rows,
            button_row,
            self.output_widget
        ])

        display(main_container)

    def on_reset_click(self, button):
        """Reset tất cả giá trị về mặc định"""
        for feature in self.feature_names:
            min_val, max_val = self.feature_ranges[feature]
            default_val = (min_val + max_val) / 2
            self.input_widgets[feature].value = round(default_val, 2)

        with self.output_widget:
            clear_output()
            print("🔄 Đã reset tất cả giá trị về mặc định!")

    def on_predict_click(self, button):
        """Xử lý khi nhấn nút dự đoán"""
        with self.output_widget:
            clear_output()

            # Lấy dữ liệu từ widgets
            data = {}
            for feature, widget in self.input_widgets.items():
                data[feature] = widget.value

            # Kiểm tra dữ liệu hợp lệ
            if self.validate_input(data):
                self.predict_and_display(data)

    def validate_input(self, data_dict):
        """Kiểm tra tính hợp lệ của dữ liệu đầu vào"""
        errors = []

        for feature, value in data_dict.items():
            min_val, max_val = self.feature_ranges[feature]
            if value < min_val * 0.5 or value > max_val * 2:  # Cho phép một chút flexibility
                errors.append(f"⚠️ {feature}: {value:.2f} (khuyến nghị: {min_val:.1f}-{max_val:.1f})")

        if errors:
            warning_html = f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4 style="color: #856404;">⚠️ Cảnh báo về giá trị đầu vào:</h4>
                <ul style="color: #856404; margin: 10px 0;">
                    {''.join([f'<li>{error}</li>' for error in errors])}
                </ul>
                <p style="color: #856404; font-style: italic;">Kết quả dự đoán có thể không chính xác với các giá trị ngoài phạm vi thông thường.</p>
            </div>
            """
            display(HTML(warning_html))

        return True

    def predict_and_display(self, data_dict):
        """Dự đoán và hiển thị kết quả"""
        if not self.lr_model or not self.nb_model:
            print("❌ Chưa tải mô hình!")
            return

        # Chuyển đổi thành DataFrame với đúng thứ tự features
        df = pd.DataFrame([data_dict])

        # Đảm bảo thứ tự columns đúng với model
        if hasattr(self.lr_model, 'feature_names_in_'):
            df = df[self.lr_model.feature_names_in_]
        else:
            df = df[self.feature_names]

        # Dự đoán
        lr_pred = self.lr_model.predict(df)[0]
        nb_pred = self.nb_model.predict(df)[0]

        lr_proba = self.lr_model.predict_proba(df)[0]
        nb_proba = self.nb_model.predict_proba(df)[0]

        lr_classes = self.lr_model.classes_
        nb_classes = self.nb_model.classes_

        # Hiển thị kết quả đẹp
        self.display_beautiful_results(data_dict, lr_pred, nb_pred,
                                     lr_proba, nb_proba, lr_classes, nb_classes)

    def display_beautiful_results(self, input_data, lr_pred, nb_pred,
                                lr_proba, nb_proba, lr_classes, nb_classes):
        """Hiển thị kết quả đẹp mắt cho wine prediction"""

        # HTML cho input data (chỉ hiển thị một số key features)
        key_features = ['Alcohol', 'Total phenols', 'Flavanoids', 'Color intensity', 'Proline']
        input_html = "<div style='background: linear-gradient(135deg, #FFF8DC, #F5DEB3); padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 5px solid #DAA520;'>"
        input_html += "<h4>🍷 Đặc trưng chính của mẫu rượu:</h4>"
        input_html += "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>"
        for feature in key_features:
            if feature in input_data:
                input_html += f"<div><strong>{feature}:</strong> {input_data[feature]:.2f}</div>"
        input_html += "</div></div>"

        # Kết quả dự đoán
        lr_confidence = max(lr_proba)
        nb_confidence = max(nb_proba)

        # Tên loại rượu
        wine_names = {0: '🍷 Class 0 (Loại 1)', 1: '🍾 Class 1 (Loại 2)', 2: '🥂 Class 2 (Loại 3)'}

        results_html = f"""
        <div style="display: flex; gap: 20px; margin: 20px 0;">
            <div style="flex: 1; background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 20px; border-radius: 15px; border-left: 5px solid #2196F3; box-shadow: 0 4px 15px rgba(33,150,243,0.3);">
                <h4>🔵 Logistic Regression</h4>
                <p><strong>🎯 Dự đoán:</strong> <span style="font-size: 20px; color: #1976D2;">{wine_names.get(lr_pred, f'Class {lr_pred}')}</span></p>
                <p><strong>🎲 Độ tin cậy:</strong> <span style="font-size: 16px; font-weight: bold;">{lr_confidence:.4f}</span></p>
                <div style="font-size: 13px; background-color: rgba(255,255,255,0.7); padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <strong>📊 Xác suất các loại rượu:</strong><br>
                    {'<br>'.join([f"• {wine_names.get(cls, f'Class {cls}')}: {prob:.4f}" for cls, prob in zip(lr_classes, lr_proba)])}
                </div>
            </div>

            <div style="flex: 1; background: linear-gradient(135deg, #e8f5e8, #c8e6c8); padding: 20px; border-radius: 15px; border-left: 5px solid #4CAF50; box-shadow: 0 4px 15px rgba(76,175,80,0.3);">
                <h4>🟢 Naive Bayes</h4>
                <p><strong>🎯 Dự đoán:</strong> <span style="font-size: 20px; color: #388E3C;">{wine_names.get(nb_pred, f'Class {nb_pred}')}</span></p>
                <p><strong>🎲 Độ tin cậy:</strong> <span style="font-size: 16px; font-weight: bold;">{nb_confidence:.4f}</span></p>
                <div style="font-size: 13px; background-color: rgba(255,255,255,0.7); padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <strong>📊 Xác suất các loại rượu:</strong><br>
                    {'<br>'.join([f"• {wine_names.get(cls, f'Class {cls}')}: {prob:.4f}" for cls, prob in zip(nb_classes, nb_proba)])}
                </div>
            </div>
        </div>
        """

        # So sánh và khuyến nghị
        if lr_pred == nb_pred:
            comparison = f"""
            <div style='background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 15px 0;'>
                <h4 style="color: #155724;">✅ Kết quả nhất quán!</h4>
                <p style="color: #155724; font-size: 16px;"><strong>Cả hai mô hình đều dự đoán:</strong> {wine_names.get(lr_pred, f'Class {lr_pred}')}</p>
            </div>
            """
        else:
            comparison = f"""
            <div style='background: linear-gradient(135deg, #fff3cd, #ffeaa7); padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 15px 0;'>
                <h4 style="color: #856404;">⚠️ Kết quả khác nhau:</h4>
                <p style="color: #856404;"><strong>Logistic Regression:</strong> {wine_names.get(lr_pred, f'Class {lr_pred}')}</p>
                <p style="color: #856404;"><strong>Naive Bayes:</strong> {wine_names.get(nb_pred, f'Class {nb_pred}')}</p>
            </div>
            """

        best_model = "Logistic Regression" if lr_confidence > nb_confidence else "Naive Bayes"
        best_confidence = max(lr_confidence, nb_confidence)
        best_pred = lr_pred if lr_confidence > nb_confidence else nb_pred

        if best_confidence > 0.9:
            confidence_level = "Rất cao 🌟"
            confidence_color = "#28a745"
        elif best_confidence > 0.7:
            confidence_level = "Cao 👍"
            confidence_color = "#17a2b8"
        elif best_confidence > 0.5:
            confidence_level = "Trung bình ⚡"
            confidence_color = "#ffc107"
        else:
            confidence_level = "Thấp ⚠️"
            confidence_color = "#dc3545"

        recommendation = f"""
        <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid {confidence_color}; box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
            <h4 style="color: #495057;">🏆 Khuyến nghị cuối cùng:</h4>
            <p><strong>🥇 Mô hình tin cậy nhất:</strong> <span style="color: {confidence_color}; font-weight: bold;">{best_model}</span></p>
            <p><strong>🍷 Loại rượu dự đoán:</strong> <span style="color: {confidence_color}; font-size: 18px; font-weight: bold;">{wine_names.get(best_pred, f'Class {best_pred}')}</span></p>
            <p><strong>📊 Độ tin cậy:</strong> <span style="color: {confidence_color}; font-weight: bold;">{best_confidence:.4f}</span></p>
            <p><strong>📈 Mức độ tin cậy:</strong> <span style="color: {confidence_color}; font-weight: bold; font-size: 16px;">{confidence_level}</span></p>
        </div>
        """

        # Hiển thị tất cả
        display(HTML(input_html + results_html + comparison + recommendation))

        # Vẽ biểu đồ xác suất
        self.plot_probabilities(lr_classes, lr_proba, nb_proba)

    def plot_probabilities(self, classes, lr_proba, nb_proba):
        """Vẽ biểu đồ so sánh xác suất cho wine prediction"""
        fig, ax = plt.subplots(figsize=(12, 7))

        wine_names = ['Class 0\n(Loại 1)', 'Class 1\n(Loại 2)', 'Class 2\n(Loại 3)']
        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax.bar(x - width/2, lr_proba, width, label='Logistic Regression',
                      color='#2196F3', alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, nb_proba, width, label='Naive Bayes',
                      color='#4CAF50', alpha=0.8, edgecolor='white', linewidth=2)

        ax.set_xlabel('Loại Rượu', fontsize=12, fontweight='bold')
        ax.set_ylabel('Xác Suất Dự Đoán', fontsize=12, fontweight='bold')
        ax.set_title('🍷 So Sánh Xác Suất Dự Đoán Loại Rượu', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(wine_names)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Thêm giá trị lên bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontweight='bold')

        plt.style.use('default')
        plt.tight_layout()
        plt.show()

# =============================================================================
# PHƯƠNG THỨC SỬ DỤNG NHANH CHO WINE PREDICTION
# =============================================================================

def create_wine_prediction_interface():
    """Tạo giao diện dự đoán loại rượu"""
    predictor = WineModelPredictor()

    # Thử tải mô hình, nếu không được thì dùng demo
    if not predictor.load_models():
        predictor.setup_demo_models()

    # Tạo giao diện
    predictor.create_input_widgets()
    predictor.display_input_interface()

    return predictor

def create_wine_sliders():
    """Tạo sliders cho wine prediction"""
    predictor = WineModelPredictor()

    if not predictor.load_models():
        predictor.setup_demo_models()

    def predict_wine(**kwargs):
        data = {k: v for k, v in kwargs.items()}
        predictor.predict_and_display(data)

    # Tạo sliders cho từng feature với ranges thực tế
    sliders = {}
    for feature in predictor.feature_names:
        min_val, max_val = predictor.feature_ranges[feature]
        default_val = (min_val + max_val) / 2

        sliders[feature] = widgets.FloatSlider(
            value=default_val,
            min=min_val * 0.8,
            max=max_val * 1.2,
            step=(max_val - min_val) / 100,
            description=feature,
            style={'description_width': '200px'},
            layout=widgets.Layout(width='500px'),
            readout_format='.2f'
        )

    return interact(predict_wine, **sliders)

# =============================================================================
# HƯỚNG DẪN SỬ DỤNG CHO WINE PREDICTION
# =============================================================================

def show_wine_usage_guide():
    """Hiển thị hướng dẫn sử dụng cho wine prediction"""
    guide_html = """
    <div style="background: linear-gradient(135deg, #722F37, #C73E1D);
                color: white; padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <h2>🍷 HƯỚNG DẪN SỬ DỤNG WINE PREDICTION</h2>

        <h3>🚀 Cách 1: Giao diện Widgets (Khuyên dùng)</h3>
        <pre style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
# Tạo giao diện dự đoán loại rượu
predictor = create_wine_prediction_interface()
        </pre>

        <h3>🎚️ Cách 2: Sliders tương tác cho 13 đặc trưng</h3>
        <pre style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
# Tạo sliders để điều chỉnh 13 đặc trưng hóa học
create_wine_sliders()
        </pre>

        <h3>🔧 Cách 3: Sử dụng thủ công</h3>
        <pre style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
# Khởi tạo thủ công
predictor = WineModelPredictor()
predictor.load_models()  # hoặc predictor.setup_demo_models()
predictor.display_input_interface()
        </pre>

        <h3>🍷 Thông tin về 13 đặc trưng:</h3>
        <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 14px;">
                <div><strong>1. Alcohol:</strong> Độ cồn (11.0-15.0%)</div>
                <div><strong>2. Malic acid:</strong> Axit malic (0.7-6.0 g/L)</div>
                <div><strong>3. Ash:</strong> Tro (1.4-3.3 g/L)</div>
                <div><strong>4. Alcalinity of ash:</strong> Độ kiềm của tro (10.6-30.0)</div>
                <div><strong>5. Magnesium:</strong> Magie (70-162 mg/L)</div>
                <div><strong>6. Total phenols:</strong> Tổng phenol (0.98-4.0)</div>
                <div><strong>7. Flavanoids:</strong> Flavanoid (0.34-5.08)</div>
                <div><strong>8. Nonflavanoid phenols:</strong> Phenol không phải flavanoid (0.13-0.66)</div>
                <div><strong>9. Proanthocyanins:</strong> Proanthocyanin (0.41-3.58)</div>
                <div><strong>10. Color intensity:</strong> Cường độ màu sắc (1.28-13.0)</div>
                <div><strong>11. Hue:</strong> Sắc độ (0.48-1.71)</div>
                <div><strong>12. OD280/OD315:</strong> Tỷ lệ quang học (1.27-4.0)</div>
                <div><strong>13. Proline:</strong> Proline (278-1680 mg/L)</div>
            </div>
        </div>

        <h3>🎯 Kết quả dự đoán:</h3>
        <ul style="font-size: 14px; line-height: 1.6;">
            <li><strong>Class 0 (Loại 1):</strong> Rượu vang loại 1</li>
            <li><strong>Class 1 (Loại 2):</strong> Rượu vang loại 2</li>
            <li><strong>Class 2 (Loại 3):</strong> Rượu vang loại 3</li>
        </ul>

        <h3>💡 Lưu ý quan trọng:</h3>
        <ul style="font-size: 14px; line-height: 1.6;">
            <li>Hệ thống sẽ cảnh báo nếu giá trị ngoài phạm vi thông thường</li>
            <li>Giao diện hiển thị so sánh giữa 2 mô hình ML</li>
            <li>Kết quả bao gồm độ tin cậy và khuyến nghị</li>
            <li>Tự động tạo biểu đồ trực quan cho xác suất</li>
            <li>Tương thích với Jupyter Notebook và Google Colab</li>
            <li>Nếu chưa có mô hình trained, sẽ tự động dùng Wine dataset demo</li>
        </ul>

        <h3>🔍 Ví dụ giá trị mẫu:</h3>
        <pre style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
# Mẫu rượu vang điển hình:
Alcohol: 13.2, Malic acid: 2.8, Ash: 2.2, Alcalinity of ash: 18.5
Magnesium: 112, Total phenols: 2.85, Flavanoids: 2.91, Nonflavanoid phenols: 0.24
Proanthocyanins: 2.61, Color intensity: 5.7, Hue: 1.04, OD280/OD315: 3.17, Proline: 1065
        </pre>
    </div>
    """
    display(HTML(guide_html))

# =============================================================================
# DEMO FUNCTION VỚI SAMPLE DATA
# =============================================================================

def demo_wine_prediction():
    """Demo nhanh với dữ liệu mẫu"""
    predictor = WineModelPredictor()

    if not predictor.load_models():
        predictor.setup_demo_models()

    # Dữ liệu mẫu
    sample_data = {
        'Alcohol': 13.2,
        'Malic acid': 2.8,
        'Ash': 2.2,
        'Alcalinity of ash': 18.5,
        'Magnesium': 112,
        'Total phenols': 2.85,
        'Flavanoids': 2.91,
        'Nonflavanoid phenols': 0.24,
        'Proanthocyanins': 2.61,
        'Color intensity': 5.7,
        'Hue': 1.04,
        'OD280/OD315 of diluted wines': 3.17,
        'Proline': 1065
    }

    print("🍷 Demo với dữ liệu mẫu:")
    print("=" * 50)
    predictor.predict_and_display(sample_data)

# =============================================================================
# QUICK START
# =============================================================================

print("🍷 Wine Prediction Interface đã sẵn sàng!")
print("=" * 50)
print("📖 show_wine_usage_guide()      # Xem hướng dẫn chi tiết")
print("🚀 create_wine_prediction_interface()  # Tạo giao diện chính")
print("🎚️ create_wine_sliders()        # Tạo sliders tương tác")
print("🎭 demo_wine_prediction()        # Xem demo nhanh")
print("=" * 50)