
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Stock Forecast AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 STOCK FORECAST AI")
st.caption("Phân tích giá cổ phiếu và dự báo 10 phiên giao dịch")

ticker = st.text_input(
    "Nhập mã cổ phiếu Việt Nam",
    value="FPT"
).strip().upper()

period = st.selectbox(
    "Khoảng dữ liệu lịch sử",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

if st.button("Phân tích và dự báo", type="primary"):
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    symbol = ticker if "." in ticker else ticker + ".VN"

    try:
        with st.spinner("Đang tải dữ liệu và phân tích..."):
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

        if df.empty:
            st.error(
                f"Không lấy được dữ liệu cho {symbol}. "
                "Hãy kiểm tra mã hoặc thử lại sau."
            )
            st.stop()

        # Xử lý cột giá đóng cửa
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna().astype(float)

        if len(close) < 60:
            st.error("Dữ liệu quá ít để xây dựng mô hình.")
            st.stop()

        # Dùng 20 phiên trước để dự đoán phiên tiếp theo
        data = pd.DataFrame({"Close": close})
        for lag in range(1, 21):
            data[f"lag_{lag}"] = data["Close"].shift(lag)

        data = data.dropna()

        X = data.drop(columns=["Close"])
        y = data["Close"]

        # Giữ thứ tự thời gian, không xáo trộn dữ liệu
        split = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Đánh giá trên giai đoạn kiểm tra
        test_pred = model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - test_pred)))

        if len(y_test) > 1:
            mape = float(
                np.mean(
                    np.abs(
                        (y_test.to_numpy() - test_pred)
                        / np.maximum(
                            np.abs(y_test.to_numpy()), 1e-8
                        )
                    )
                ) * 100
            )
        else:
            mape = np.nan

        # Dự báo đệ quy 10 phiên tiếp theo
        history = list(close.to_numpy())
        predictions = []

        for _ in range(10):
            features = np.array(history[-20:][::-1]).reshape(1, -1)
            next_price = float(model.predict(features)[0])
            predictions.append(max(0, next_price))
            history.append(max(0, next_price))

        # Tạo ngày giao dịch dự kiến; chưa tính ngày nghỉ lễ
        last_date = pd.Timestamp(close.index[-1])
        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1),
            periods=10
        )

        forecast = pd.DataFrame({
            "Ngày dự kiến": future_dates.strftime("%d/%m/%Y"),
            "Giá dự báo": predictions
        })

        st.subheader(f"Kết quả: {symbol}")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Giá đóng cửa gần nhất",
            f"{close.iloc[-1]:,.0f}"
        )
        c2.metric(
            "Dự báo phiên thứ 10",
            f"{predictions[-1]:,.0f}"
        )
        c3.metric(
            "MAE kiểm tra",
            f"{mae:,.2f}"
        )

        st.caption(
            f"MAPE kiểm tra: {mape:.2f}%"
            if np.isfinite(mape)
            else "Không đủ dữ liệu để tính MAPE."
        )

        # Biểu đồ lịch sử và dự báo
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=close.index,
            y=close.values,
            name="Giá lịch sử",
            mode="lines"
        ))

        fig.add_trace(go.Scatter(
            x=[close.index[-1], *future_dates],
            y=[float(close.iloc[-1]), *predictions],
            name="Giá dự báo",
            mode="lines+markers",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            title="Giá đóng cửa và dự báo 10 phiên tiếp theo",
            xaxis_title="Ngày",
            yaxis_title="Giá (đơn vị theo nguồn dữ liệu)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bảng dự báo")
        st.dataframe(
            forecast.style.format({"Giá dự báo": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        csv = forecast.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Tải bảng dự báo CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

        st.warning(
            "Đây là kết quả mô hình thử nghiệm, không phải cam kết "
            "về giá tương lai hay khuyến nghị mua bán. "
            "Ngày dự kiến chưa loại trừ ngày nghỉ lễ; "
            "hãy kiểm tra ngày giao dịch thực tế."
        )

    except Exception as e:
        st.error(f"Không thể hoàn tất phân tích: {e}")
